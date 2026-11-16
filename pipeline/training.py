"""Portable fitting for three calibrated teachers and S0/S1/S2 students."""

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from data_prep import load_prepared_role
from .model import (
    TRAITS, _predict_tokens, collate, new_model, save_scoring_model, tokenize_rows,
)

TEACHER_SEEDS = (2006, 7, 24)
STUDENT_SEEDS = (4486, 4487, 4488)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_pseudo(data_dir, pseudo_path, manifest_digest):
    path = Path(pseudo_path)
    if path.is_dir():
        path = path / "pseudo.jsonl"
    manifest = json.loads((path.parent / "pseudo_manifest.json").read_text(encoding="utf-8"))
    external = load_prepared_role(data_dir, "external")
    if (manifest.get("schema") != "ell_teacher_pseudo_v1"
            or manifest.get("teacher_seeds") != list(TEACHER_SEEDS)
            or manifest.get("data_manifest_sha256") != manifest_digest
            or manifest["sha256"]["pseudo.jsonl"] != _sha256(path)
            or manifest["sha256"]["external.jsonl"] != _sha256(data_dir / "external.jsonl")):
        raise ValueError("Pseudo labels do not match the three-teacher cache and prepared sample")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != len(external):
        raise ValueError("Pseudo-label coverage differs from the prepared sample")
    for row, origin in zip(rows, external, strict=True):
        if (row["text_id"], row["full_text"]) != (origin["text_id"], origin["full_text"]):
            raise ValueError("Pseudo labels must match every prepared external row in order")
        for field in ("target_mean", "weight"):
            vector = np.asarray(row[field], dtype=np.float64)
            if vector.shape != (len(TRAITS),) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid pseudo-label vector: {row['text_id']} {field}")
            if field == "weight" and (vector < 0).any():
                raise ValueError(f"Negative pseudo-label weight: {row['text_id']}")
    return rows


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _optimizer(model, teacher):
    top_rate = 1e-5 if teacher else 5e-6
    groups = {}
    top_group = (len(model.backbone.encoder.layer) - 1) // 4
    for name, parameter in model.named_parameters():
        if name.startswith("variance_head."):
            parameter.requires_grad_(False)
            continue
        if name.startswith("score_head."):
            rate = top_rate * 10
        elif name.startswith("backbone.embeddings."):
            rate = top_rate / 2
        elif name.startswith("backbone.encoder.layer."):
            group = int(name.split(".")[3]) // 4
            rate = top_rate * 0.95 ** (top_group - group)
        else:
            rate = top_rate
        decay = 0.0 if parameter.ndim <= 1 else 0.01
        groups.setdefault((rate, decay), []).append(parameter)
    return torch.optim.AdamW([
        {"params": params, "lr": rate, "base_lr": rate, "weight_decay": decay}
        for (rate, decay), params in groups.items()
    ], eps=1e-6, foreach=False)


def _balanced_sizes(count, parts):
    quotient, remainder = divmod(count, parts)
    return [quotient + 1] * remainder + [quotient] * (parts - remainder)


def _epoch_windows(gold_count, pseudo_count, batch, micro, seed, epoch):
    gold = list(range(gold_count))
    random.Random(seed + 100003 * epoch).shuffle(gold)
    if not pseudo_count:
        return [[(False, index) for index in gold[start:start + batch]]
                for start in range(0, gold_count, batch)]

    window_count = math.ceil((gold_count + pseudo_count) / batch)
    if window_count > min(gold_count, pseudo_count):
        raise ValueError("Increase the effective batch size so every mixed update uses both domains")
    pseudo = list(range(pseudo_count))
    random.Random(seed + 7000003 + 100003 * epoch).shuffle(pseudo)
    layout = random.Random(seed + 14000027 + 100003 * epoch)
    gold_sizes = _balanced_sizes(gold_count, window_count)
    totals = _balanced_sizes(gold_count + pseudo_count, window_count)
    # Randomize complete windows, carrying their domain counts together.
    sizes = list(zip(gold_sizes, totals, strict=True))
    layout.shuffle(sizes)
    windows = []
    gold_cursor = pseudo_cursor = 0
    for ng, total in sizes:
        npseudo = total - ng
        if not 0 < ng < total:
            raise ValueError("Invalid mixed optimizer window")
        parts = math.ceil(total / micro)
        if total % micro == 0 and min(ng, npseudo) >= parts:
            micro_gold = _balanced_sizes(ng, parts)
            layout.shuffle(micro_gold)
            domains = []
            for n in micro_gold:
                micro_domains = [False] * n + [True] * (micro - n)
                layout.shuffle(micro_domains)
                domains.extend(micro_domains)
        else:
            domains = [False] * ng + [True] * npseudo
            layout.shuffle(domains)
        window = []
        for is_pseudo in domains:
            if is_pseudo:
                window.append((True, pseudo[pseudo_cursor]))
                pseudo_cursor += 1
            else:
                window.append((False, gold[gold_cursor]))
                gold_cursor += 1
        windows.append(window)
    assert gold_cursor == gold_count and pseudo_cursor == pseudo_count
    return windows


def _autocast(device):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())


def _update(model, optimizer, window, gold, pseudo, *, micro_batch, pad_id, device, weighted):
    total_gold = sum(not item[0] for item in window)
    total_pseudo = len(window) - total_gold
    optimizer.zero_grad(set_to_none=True)
    running = torch.zeros(2, device=device)
    for offset in range(0, len(window), micro_batch):
        block = window[offset:offset + micro_batch]
        rows = [pseudo[index] if is_pseudo else gold[index] for is_pseudo, index in block]
        inputs = [tensor.to(device) for tensor in collate(rows, pad_id)]
        mask = torch.tensor([is_pseudo for is_pseudo, _ in block], device=device, dtype=torch.bool)
        targets = torch.tensor(
            [row["target_mean"] if is_pseudo else row["labels"]
             for (is_pseudo, _), row in zip(block, rows, strict=True)],
            dtype=torch.float32, device=device,
        )
        with _autocast(device):
            predictions = model(*inputs)
        squared = (predictions.float() - targets).square()
        gold_loss = squared[~mask].sum() / (total_gold * len(TRAITS))
        if total_pseudo:
            weights = torch.tensor(
                [row["weight"] if weighted and is_pseudo else [1.0] * len(TRAITS)
                 for (is_pseudo, _), row in zip(block, rows, strict=True)],
                dtype=torch.float32, device=device,
            )
            pseudo_loss = (weights[mask] * squared[mask]).sum() / (total_pseudo * len(TRAITS))
        else:
            pseudo_loss = squared.new_zeros(())
        (gold_loss + 0.5 * pseudo_loss).backward()
        running += torch.stack((gold_loss.detach(), pseudo_loss.detach()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return running.cpu().numpy()


def _validation(model, rows, pad_id, device, batch_size):
    predictions = _predict_tokens(model, rows, pad_id, device, batch_size)
    targets = np.asarray([row["labels"] for row in rows], dtype=np.float64)
    trait_rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    return float(trait_rmse.mean()), dict(zip(TRAITS, map(float, trait_rmse), strict=True))


def _fit_scores(model, tokenizer, gold, validation, pseudo, run_dir, device, *,
                teacher, seed, family, epochs, effective_batch, micro_batch):
    pad_id = tokenizer.pad_token_id
    optimizer = _optimizer(model, teacher)
    window_count = math.ceil((len(gold) + len(pseudo)) / effective_batch)
    updates = epochs * window_count
    warmup = math.ceil(updates * 0.06)
    update = 0
    best, stale, selected = math.inf, 0, None
    history = []
    with TemporaryDirectory(prefix=".fit-", dir=run_dir) as temporary:
        best_path = Path(temporary) / "best.pt"
        for epoch in range(1, epochs + 1):
            windows = _epoch_windows(len(gold), len(pseudo), effective_batch, micro_batch, seed, epoch)
            quarters = {math.ceil(window_count * q / 4) for q in range(1, 5)}
            epoch_losses = []
            for step, window in enumerate(windows, start=1):
                factor = ((update + 1) / warmup if update < warmup
                          else 0.5 * (1 + math.cos(math.pi * (update - warmup)
                                                  / max(1, updates - warmup))))
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * factor
                model.train()
                losses = _update(model, optimizer, window, gold, pseudo, micro_batch=micro_batch,
                                 pad_id=pad_id, device=device, weighted=family == "S2")
                epoch_losses.append(losses)
                update += 1
                if step != window_count and (not teacher or step not in quarters):
                    continue
                value, by_trait = _validation(model, validation, pad_id, device, min(8, micro_batch))
                seen = np.mean(epoch_losses, axis=0)
                entry = {
                    "epoch": epoch - 1 + step / window_count, "optimizer_steps": update,
                    "gold_mse": float(seen[0]), "pseudo_mse": float(seen[1]),
                    "validation_mcrmse": value, "validation_rmse": by_trait,
                }
                history.append(entry)
                _write_json(run_dir / "history.json", history)
                print(f"{family} seed={seed} epoch={entry['epoch']:.2f} validation_mcrmse={value:.6f}",
                      flush=True)
                if value < best:
                    best, stale, selected = value, 0, entry
                    torch.save(model.state_dict(), best_path)
                else:
                    stale += 1
            if not teacher and stale >= 2:
                break
        del optimizer
        try:
            model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True), strict=True)
        except FileNotFoundError as exc:
            raise FloatingPointError("No finite validation checkpoint was selected") from exc
    return selected, history


def _calibrate(model, tokenizer, calibration, validation, device, batch_size):
    pad_id = tokenizer.pad_token_id

    def cache(rows):
        features, predictions = _predict_tokens(model, rows, pad_id, device, batch_size, features=True)
        targets = torch.tensor([row["labels"] for row in rows], dtype=torch.float32)
        return torch.from_numpy(features), (torch.from_numpy(predictions) - targets).square()

    train_features, train_error = cache(calibration)
    val_features, val_error = cache(validation)
    head = model.variance_head.to("cpu").requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": [head.weight], "weight_decay": 0.01},
        {"params": [head.bias], "weight_decay": 0.0},
    ], lr=0.001, eps=1e-6, foreach=False)

    def nll(features, error):
        variance = 0.01 + F.softplus(head(features))
        return 0.5 * (variance.log() + error / variance).mean()

    with torch.no_grad():
        best = float(nll(val_features, val_error))
    best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    best_epoch = 0
    history = [{"epoch": 0, "validation_nll": best}]
    for epoch in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        loss = nll(train_features, train_error)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            value = float(nll(val_features, val_error))
        if not math.isfinite(value):
            raise FloatingPointError("Calibration validation NLL must be finite")
        history.append({"epoch": epoch, "validation_nll": value})
        if value < best:
            best, best_epoch = value, epoch
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        elif epoch - best_epoch >= 20:
            break
    head.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation_nll": best, "history": history}


def _train(data_dir, model_path, run_dir, seed, family, *, pseudo_path, device,
           micro_batch, effective_batch, epochs, max_length):
    if not 1 <= micro_batch <= effective_batch or epochs < 1 or not 3 <= max_length <= 1024:
        raise ValueError("Check the batch sizes, epoch count, and maximum input length")
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Use a new output directory: {run_dir}")
    teacher = family == "T"
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Choose CPU or an available CUDA GPU")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest_digest = _sha256(data_dir / "manifest.json")
    gold = load_prepared_role(data_dir, "teacher_score_train" if teacher else "gold_train")
    validation = load_prepared_role(data_dir, "gold_val")
    calibration = load_prepared_role(data_dir, "teacher_calibration") if teacher else None
    gold_digests = {role: _sha256(data_dir / f"{role}.jsonl") for role in
                    ("teacher_score_train", "teacher_calibration", "gold_val")} if teacher else None
    if family in ("S1", "S2"):
        if pseudo_path is None:
            raise ValueError("S1/S2 require the freshly generated pseudo-label cache")
        pseudo = _read_pseudo(data_dir, pseudo_path, manifest_digest)
    else:
        if pseudo_path is not None:
            raise ValueError("Teachers and S0 train on human scores")
        pseudo = []
    if (_sha256(data_dir / "manifest.json") != manifest_digest
            or not gold or not validation or teacher and not calibration
            or family in ("S1", "S2") and not pseudo):
        raise ValueError("Prepared split is incomplete or changed while loading")
    means = np.asarray([row["labels"] for row in gold], dtype=np.float64).mean(axis=0)
    _seed(seed)
    model, tokenizer = new_model(model_path, means, teacher=teacher, max_length=max_length)
    if device.type == "cuda":
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    gold = tokenize_rows(gold, tokenizer, max_length)
    validation = tokenize_rows(validation, tokenizer, max_length)
    pseudo = tokenize_rows(pseudo, tokenizer, max_length)
    run_dir.mkdir(parents=True)
    selected, history = _fit_scores(
        model, tokenizer, gold, validation, pseudo, run_dir, device, teacher=teacher, seed=seed, family=family,
        epochs=epochs, effective_batch=effective_batch, micro_batch=micro_batch,
    )
    result = {"family": family, "seed": seed, "best_epoch": selected["epoch"],
              "validation_mcrmse": selected["validation_mcrmse"],
              "validation_rmse": selected["validation_rmse"], "data_manifest_sha256": manifest_digest,
              "scoring_validations": len(history), "lambda": 0.5 if pseudo else 0.0}
    if teacher:
        result["gold_role_sha256"] = gold_digests
        var = _calibrate(model, tokenizer, tokenize_rows(calibration, tokenizer, max_length),
                         validation, device, min(8, micro_batch))
        _write_json(run_dir / "variance_history.json", var.pop("history"))
        result["variance_calibration"] = var
    save_scoring_model(run_dir / "checkpoint", model.to("cpu"), tokenizer, result)
    _write_json(run_dir / "result.json", result)
    return result


def train_teacher(data_dir, model_path, run_dir, seed, *, device="cuda", micro_batch=16,
                  effective_batch=32, epochs=3, max_length=1024):
    if seed not in TEACHER_SEEDS:
        raise ValueError(f"Teacher seed must be one of {TEACHER_SEEDS}")
    return _train(data_dir, model_path, run_dir, seed, "T", pseudo_path=None, device=device,
                  micro_batch=micro_batch, effective_batch=effective_batch, epochs=epochs, max_length=max_length)


def train_student(data_dir, model_path, run_dir, seed, family, *, pseudo_path=None, device="cuda",
                  micro_batch=4, effective_batch=32, epochs=6, max_length=1024):
    if family not in ("S0", "S1", "S2") or seed not in STUDENT_SEEDS:
        raise ValueError("Use S0/S1/S2 and seed 4486, 4487, or 4488")
    return _train(data_dir, model_path, run_dir, seed, family, pseudo_path=pseudo_path, device=device,
                  micro_batch=micro_batch, effective_batch=effective_batch, epochs=epochs, max_length=max_length)
