"""Label an external pool with calibrated teachers and predict six-trait CSVs."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

TRAITS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
TEACHER_SEEDS = (2006, 7, 24)
STUDENT_SEEDS = (4486, 4487, 4488)


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checkpoints(directories):
    checkpoints = []
    for directory in directories:
        directory = Path(directory).resolve()
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
        if (metadata.get("schema") != "ell_scoring_checkpoint_v1"
                or metadata.get("traits") != list(TRAITS)
                or metadata.get("family") not in {"T", "S0", "S1", "S2"}
                or type(metadata.get("seed")) is not int):
            raise ValueError(f"Invalid scoring checkpoint metadata in {directory}")
        checkpoints.append((directory, metadata))
    return checkpoints


def aggregate_teachers(means, variances, tau2=0.04):
    """Aggregate exactly three teacher distributions in FP32."""
    import torch

    means = torch.as_tensor(means, dtype=torch.float32)
    variances = torch.as_tensor(variances, dtype=torch.float32)
    if (means.ndim != 3 or means.shape[0] != 3 or means.shape[-1] != len(TRAITS)
            or means.shape != variances.shape or means.shape[1] == 0
            or not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(variances).all())
            or bool((variances <= 0).any()) or not math.isfinite(tau2) or tau2 <= 0):
        raise ValueError("Expected three aligned finite [K,N,6] means and positive variances")

    raw_mean = means.mean(0)
    aleatoric = variances.mean(0)
    epistemic = (means - raw_mean.unsqueeze(0)).square().mean(0)
    uncertainty = aleatoric + epistemic
    confidence = tau2 / (tau2 + uncertainty)
    relative = confidence / (confidence.mean(0, keepdim=True) + 1e-8)
    capped = relative.clamp(max=3.0)
    weights = capped / (capped.mean(0, keepdim=True) + 1e-8)
    return {
        "raw_mean": raw_mean, "target_mean": raw_mean.clamp(1, 5),
        "aleatoric": aleatoric, "epistemic": epistemic, "uncertainty": uncertainty,
        "confidence": confidence, "relative_confidence": relative,
        "capped_relative_confidence": capped, "weight": weights,
    }


def generate_pseudo_labels(data_dir, teacher_dirs, output_dir, *, device="cpu", batch_size=8):
    """Create a new pseudo-label directory bound to the prepared external pool."""
    from data_prep import load_prepared_role, read_prepared_manifest
    from .model import load_scoring_model, predict_rows

    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new pseudo-label directory: {output_dir}")
    manifest_path = data_dir / "manifest.json"
    manifest_sha = _sha256(manifest_path)
    prepared = read_prepared_manifest(data_dir)
    external = load_prepared_role(data_dir, "external")
    if not external:
        raise ValueError("Run data preparation with --earlier-dir to create an external essay pool")
    external_entry = prepared["files"]["external"]
    external_path = data_dir / external_entry["path"]
    teachers = _checkpoints(teacher_dirs)
    by_seed = {metadata["seed"]: (directory, metadata) for directory, metadata in teachers}
    if len(teachers) != 3 or set(by_seed) != set(TEACHER_SEEDS):
        raise ValueError(f"Expected calibrated teachers for seeds {TEACHER_SEEDS}")
    gold_roles = ("teacher_score_train", "teacher_calibration", "gold_val")
    gold_role_hashes = {role: prepared["files"][role]["sha256"] for role in gold_roles}
    for directory, metadata in teachers:
        recorded_roles = metadata.get("gold_role_sha256")
        same_gold = (recorded_roles == gold_role_hashes if recorded_roles is not None
                     else metadata.get("data_manifest_sha256") == manifest_sha)
        if metadata["family"] != "T" or metadata.get("has_variance") is not True or not same_gold:
            raise ValueError(f"Teacher was not calibrated on these prepared gold splits: {directory}")

    means, variances, teacher_sources = [], [], []
    for seed in TEACHER_SEEDS:
        directory, _ = by_seed[seed]
        model, tokenizer = load_scoring_model(directory, device)
        mean, variance = predict_rows(
            model, tokenizer, external, device=device, batch_size=batch_size, with_variance=True
        )
        del model, tokenizer
        means.append(mean)
        variances.append(variance)
        teacher_sources.append({
            "seed": seed,
            "files": {
                path.relative_to(directory).as_posix(): _sha256(path)
                for path in sorted(directory.rglob("*")) if path.is_file()
            },
        })
    fields = aggregate_teachers(np.stack(means), np.stack(variances))
    if _sha256(manifest_path) != manifest_sha or _sha256(external_path) != external_entry["sha256"]:
        raise ValueError("Prepared data changed while generating pseudo-labels")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pseudo-", dir=output_dir.parent) as temporary:
        candidate = Path(temporary) / "cache"
        candidate.mkdir()
        pseudo_path = candidate / "pseudo.jsonl"
        with pseudo_path.open("w", encoding="utf-8", newline="\n") as stream:
            for index, row in enumerate(external):
                payload = {"text_id": row["text_id"], "full_text": row["full_text"]}
                payload.update({name: tensor[index].tolist() for name, tensor in fields.items()})
                stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        receipt = {
            "schema": "ell_teacher_pseudo_v1",
            "teacher_seeds": list(TEACHER_SEEDS),
            "traits": list(TRAITS),
            "records": len(external),
            "data_manifest_sha256": manifest_sha,
            "sha256": {"external.jsonl": external_entry["sha256"], "pseudo.jsonl": _sha256(pseudo_path)},
            "teacher_checkpoints": teacher_sources,
        }
        (candidate / "pseudo_manifest.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        candidate.rename(output_dir)
    return output_dir / "pseudo.jsonl"


def _read_essays(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text_id", "full_text"}.issubset(reader.fieldnames or ()):
            raise ValueError("Expected an input CSV with text_id and full_text columns")
        rows = [{key: row[key] for key in ("text_id", "full_text")} for row in reader]
    if not rows or any(not row["text_id"] or not (row["full_text"] or "").strip() for row in rows):
        raise ValueError("Input CSV must have nonempty IDs and essays")
    if len({row["text_id"] for row in rows}) != len(rows):
        raise ValueError("Input CSV contains duplicate essay IDs")
    return rows


def predict_csv(input_csv, checkpoint_dirs, output_csv, *, device="cpu", batch_size=8):
    """Predict a CSV from one checkpoint or a three-seed S1 or S2 ensemble."""
    from .model import load_scoring_model, predict_rows

    rows = _read_essays(input_csv)
    checkpoints = _checkpoints(checkpoint_dirs)
    if len(checkpoints) != 1:
        families = {metadata["family"] for _, metadata in checkpoints}
        seeds = {metadata["seed"] for _, metadata in checkpoints}
        if (len(checkpoints) != 3 or seeds != set(STUDENT_SEEDS)
                or families not in ({"S1"}, {"S2"})):
            raise ValueError("An ensemble requires one checkpoint per final seed from one student family")
        checkpoints.sort(key=lambda member: member[1]["seed"])

    total = np.zeros((len(rows), len(TRAITS)), dtype=np.float64)
    for directory, _ in checkpoints:
        model, tokenizer = load_scoring_model(directory, device)
        values = np.asarray(predict_rows(model, tokenizer, rows, device=device, batch_size=batch_size))
        del model, tokenizer
        if values.shape != total.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid scores from {directory}")
        total += values
    total /= len(checkpoints)

    output_csv = Path(output_csv)
    resolved_output = output_csv.resolve()
    if resolved_output == Path(input_csv).resolve() or any(
        resolved_output.is_relative_to(directory) for directory, _ in checkpoints
    ):
        raise ValueError("Write predictions outside the input CSV and checkpoint directories")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".predict-", dir=output_csv.parent) as temporary:
        candidate = Path(temporary) / output_csv.name
        with candidate.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("text_id", *TRAITS))
            for row, values in zip(rows, total, strict=True):
                writer.writerow((row["text_id"], *values))
        os.replace(candidate, output_csv)
    return output_csv
