"""Offline FP32 inference for historical three-seed Kaggle checkpoint bundles.

Average predictions from seeds 4486, 4487, and 4488 with equal weights.
The ensemble manifest identifies the three checkpoint bundles and their metadata.
"""
from __future__ import annotations
import argparse
import base64
import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
TRAITS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
FORMAT = "ell_deberta_offline_submission_v1"


def read_bundle(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
        keys = stream.keys()
    if metadata.get("format") != FORMAT:
        raise ValueError("Expected a self-contained ELL submission checkpoint")
    if any(not key.startswith(("backbone.", "score_head.")) for key in keys):
        raise ValueError("Checkpoint contains modules outside the scoring model")
    info = json.loads(metadata["model_info"])
    config = json.loads(metadata["config_json"])
    if info["traits"] != list(TRAITS) or info["max_length"] != 1024:
        raise ValueError("Unexpected trait order or input length")
    if config["model_type"] != "deberta-v2" or config.get("position_biased_input") is not False:
        raise ValueError("Unexpected DeBERTa relative-position configuration")
    return metadata, info, config


def load_tokenizer(metadata):
    """Restore the serialized tokenizers backend without HF class reconstruction."""
    from tokenizers import Tokenizer

    assets = base64.b64decode(metadata["tokenizer_zip_base64"], validate=True)
    if hashlib.sha256(assets).hexdigest() != metadata["tokenizer_zip_sha256"]:
        raise ValueError("Embedded tokenizer archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(assets)) as archive:
        backend = Tokenizer.from_str(archive.read("tokenizer.json").decode("utf-8"))
    backend.no_padding()
    backend.no_truncation()

    class FrozenTokenizer:
        is_fast = True
        padding_side = "right"

        def __init__(self):
            self.cls_token_id = backend.token_to_id("[CLS]")
            self.sep_token_id = backend.token_to_id("[SEP]")
            self.pad_token_id = backend.token_to_id("[PAD]")
            if any(type(value) is not int for value in
                   (self.cls_token_id, self.sep_token_id, self.pad_token_id)):
                raise ValueError("Embedded tokenizer lacks CLS/SEP/PAD IDs")

        def __call__(self, text, add_special_tokens=True, **kwargs):
            if kwargs.get("truncation") or kwargs.get("padding"):
                raise ValueError("Use the fixed external 1024-token truncation and padding")
            if isinstance(text, str):
                values = backend.encode(text, add_special_tokens=add_special_tokens).ids
            else:
                values = [item.ids for item in backend.encode_batch(
                    list(text), add_special_tokens=add_special_tokens)]
            return {"input_ids": values}

    tokenizer = FrozenTokenizer()
    probe = "The quick brown fox jumps over the lazy dog."
    body = tokenizer(probe, add_special_tokens=False)["input_ids"]
    wrapped = tokenizer(probe, add_special_tokens=True)["input_ids"]
    if wrapped != [tokenizer.cls_token_id, *body, tokenizer.sep_token_id]:
        raise ValueError("Unexpected CLS/body/SEP layout")
    return tokenizer


def load_model(path, parameters, device):
    import torch
    from safetensors.torch import load_file
    from transformers import DebertaV2Config, DebertaV2Model

    config = DebertaV2Config.from_dict(parameters)
    config._attn_implementation = "eager"
    config.output_hidden_states = False
    config.return_dict = True
    backbone = DebertaV2Model(config).float()
    head = torch.nn.Linear(config.hidden_size, len(TRAITS)).float()
    state = load_file(str(path), device="cpu")
    backbone.load_state_dict({key[len("backbone."):]: value for key, value in state.items()
                              if key.startswith("backbone.")}, strict=True)
    head.load_state_dict({key[len("score_head."):]: value for key, value in state.items()
                         if key.startswith("score_head.")}, strict=True)
    del state
    gc.collect()
    return backbone.eval().to(device), head.eval().to(device)


def read_inputs(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text_id", "full_text"}.issubset(reader.fieldnames or ()):
            raise ValueError("Test CSV must contain text_id and full_text")
        rows = [{"text_id": row["text_id"], "full_text": row["full_text"]} for row in reader]
    if not rows or any(not row["text_id"] or not isinstance(row["full_text"], str)
                       or not row["full_text"].strip() for row in rows):
        raise ValueError("Test CSV contains empty IDs/texts or has no essays")
    if len({row["text_id"] for row in rows}) != len(rows):
        raise ValueError("Test IDs must be unique")
    return rows


def resolve_test_path(test=None, input_root=Path("/kaggle/input")):
    if test is not None:
        return Path(test)
    candidates = sorted(path for path in Path(input_root).rglob("test.csv")
                        if "feedback-prize-english-language-learning" in path.parts
                        and path.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one competition test.csv under {input_root}, found {len(candidates)}: "
            f"{candidates}. Pass --test explicitly to select a file.")
    return candidates[0]


def encode(tokenizer, rows):
    body_ids = tokenizer([row["full_text"] for row in rows], add_special_tokens=False,
                         truncation=False, return_attention_mask=False)["input_ids"]
    return [[tokenizer.cls_token_id, *body[:1022], tokenizer.sep_token_id] for body in body_ids]


def score_hidden(hidden, mask, head):
    # Includes CLS/SEP; excludes only right padding. Training dropout is off in eval.
    weights = mask.float().unsqueeze(-1)
    pooled = (hidden.float() * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    return head(pooled)


def predict(backbone, head, encoded, pad_id, batch_size, device):
    import torch

    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
    predictions = [None] * len(encoded)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            width = len(encoded[indices[-1]])
            ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for position, index in enumerate(indices):
                tokens = encoded[index]
                ids[position, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                mask[position, :len(tokens)] = 1
            ids, mask = ids.to(device), mask.to(device)
            hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            scores = score_hidden(hidden, mask, head)
            if scores.shape != (len(indices), len(TRAITS)) or not bool(torch.isfinite(scores).all()):
                raise RuntimeError("Malformed/nonfinite model predictions")
            for index, values in zip(indices, scores.cpu().tolist()):
                predictions[index] = values
            del ids, mask, hidden, scores
            print(f"Predicted {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return predictions


def write_submission(output, rows, scores, sample_submission=None):
    by_id = {row["text_id"]: values for row, values in zip(rows, scores)}
    identifiers = [row["text_id"] for row in rows]
    if sample_submission is not None:
        with Path(sample_submission).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["text_id", *TRAITS]:
                raise ValueError("Unexpected sample_submission columns")
            identifiers = [row["text_id"] for row in reader]
        if len(identifiers) != len(by_id) or set(identifiers) != set(by_id):
            raise ValueError("Test IDs differ from sample_submission IDs")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["text_id", *TRAITS])
        for identifier in identifiers:
            writer.writerow([identifier, *by_id[identifier]])
    return output


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("S1", "S2"), required=True)
    parser.add_argument("--manifest", type=Path, help="Historical ensemble manifest JSON")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"),
                        help="Directory containing the historical checkpoint bundles")
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    require(args.batch_size > 0 and args.cpu_threads > 0, "Invalid batch size/CPU threads")
    manifest_path = args.manifest or Path(__file__).resolve().parent / f"ensemble_manifest_{args.family}.json"
    if not manifest_path.is_file():
        parser.error(
            f"Ensemble manifest is missing: {manifest_path}. Supply the historical manifest "
            "with --manifest and its checkpoint bundles under --input-root. For checkpoints "
            "created by this repository, use python predict.py --checkpoints <checkpoint1> "
            "<checkpoint2> <checkpoint3> --input <csv> --output <csv>."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    require(manifest.get("schema") == "ell_three_seed_prediction_ensemble_v1"
            and manifest.get("status") == "complete" and not manifest.get("pending_models"),
            "All three final best checkpoints must be complete; partial ensemble execution is forbidden")
    require(manifest["family"] == args.family and manifest["seeds"] == [4486, 4487, 4488]
            and manifest["traits"] == list(TRAITS) and manifest["lambda_pseudo"] == 0.5
            and manifest["max_length"] == 1024 and manifest["precision"] == "FP32"
            and manifest["parameter_averaging"] is False
            and manifest["clipping"] is False and manifest["rounding"] is False,
            "Unexpected ensemble definition")
    members = manifest["checkpoints"]
    require(len(members) == 3, "Exactly three checkpoints are required")
    paths = []
    for seed, member in zip([4486, 4487, 4488], members):
        require(member["model_info"]["seed"] == seed
                and member["model_info"]["model_id"] == f"{args.family}_l05_{seed}"
                and math.isclose(member["weight"], 1 / 3, abs_tol=1e-15), "Member seed/name/weight mismatch")
        filename = member["filename"]
        require(Path(filename).name == filename, "Checkpoint manifest requires a basename")
        matches = sorted(p for p in args.input_root.rglob(filename) if p.is_file())
        require(len(matches) == 1, f"Expected one {filename} under {args.input_root}, found {matches}")
        path = matches[0]
        require(path.stat().st_size == member["bytes"] and hash_file(path) == member["sha256"],
                f"Checkpoint size/checksum mismatch: {path}")
        paths.append(path)
    import torch
    torch.set_num_threads(args.cpu_threads)
    require(args.device != "cuda" or torch.cuda.is_available(), "Enable a Kaggle GPU accelerator")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    test = resolve_test_path(args.test, args.input_root)
    rows = read_inputs(test)
    totals = [[0.0] * len(TRAITS) for _ in rows]
    encoded, tokenizer, reference_config, reference_tokenizer = None, None, None, None
    for path, member in zip(paths, members):
        metadata, info, parameters = read_bundle(path)
        require(info == member["model_info"] and info["precision"] == "FP32"
                and info["lambda_pseudo"] == 0.5,
                f"Embedded checkpoint identity differs from manifest: {path}")
        if encoded is None:
            reference_config, reference_tokenizer = metadata["config_json"], metadata["tokenizer_zip_sha256"]
            tokenizer = load_tokenizer(metadata)
            encoded = encode(tokenizer, rows)
        else:
            require(metadata["config_json"] == reference_config
                    and metadata["tokenizer_zip_sha256"] == reference_tokenizer,
                    "Members use different configs/tokenizers")
        print(json.dumps({"event": "predicting_member", "model_id": info["model_id"], "seed": info["seed"],
                          "selected_epoch": info["selected_epoch"], "precision": "fp32", "max_length": 1024}), flush=True)
        backbone, head = load_model(path, parameters, device)
        predictions = predict(backbone, head, encoded, tokenizer.pad_token_id, args.batch_size, device)
        require(len(predictions) == len(rows), "Member prediction coverage mismatch")
        by_id = {row["text_id"]: values for row, values in zip(rows, predictions)}
        require(len(by_id) == len(rows), "Member prediction IDs are not unique")
        for index, row in enumerate(rows):
            values = by_id[row["text_id"]]
            require(len(values) == 6 and all(math.isfinite(x) for x in values), "Malformed member prediction")
            for trait, value in enumerate(values):
                totals[index][trait] += float(value)
        del predictions, by_id, backbone, head
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean = [[value / 3 for value in row] for row in totals]
    sample = test.parent / "sample_submission.csv"
    output = write_submission(args.output, rows, mean, sample if sample.is_file() else None)
    print(json.dumps({"event": "ensemble_saved", "ensemble_id": manifest["ensemble_id"],
                      "models": 3, "seeds": [4486, 4487, 4488], "rows": len(rows), "output": str(output),
                      "precision": "fp32", "mean_accumulation": "CPU float64", "clipping": False,
                      "rounding": False, "parameter_averaging": False}), flush=True)
    return output


if __name__ == "__main__":
    main()
