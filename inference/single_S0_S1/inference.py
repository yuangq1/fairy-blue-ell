"""FP32 single-model inference for S0 and S1 at seed 4486."""
from __future__ import annotations

import argparse
import base64
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TRAITS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
CHECKPOINTS = {
    "S0": "S0_seed4486_best_e5.safetensors",
    "S1_l05": "S1_lambda0p5_seed4486_best_e3.safetensors",
}
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), default="S0")
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--test", type=Path, default=Path("/kaggle/input/feedback-prize-english-language-learning/test.csv"))
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("Batch size and CPU threads must be positive")
    import torch

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; enable a Kaggle GPU accelerator or select --device cpu")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint = args.bundle_dir / CHECKPOINTS[args.model]
    metadata, info, parameters = read_bundle(checkpoint)
    print(json.dumps({"model": args.model, "selected_epoch": info["selected_epoch"],
                      "max_length": 1024, "precision": "fp32", "device": str(device)}), flush=True)
    tokenizer = load_tokenizer(metadata)
    rows = read_inputs(args.test)
    encoded = encode(tokenizer, rows)
    backbone, head = load_model(checkpoint, parameters, device)
    scores = predict(backbone, head, encoded, tokenizer.pad_token_id, args.batch_size, device)
    sample = args.sample_submission
    if sample is None and (args.test.parent / "sample_submission.csv").is_file():
        sample = args.test.parent / "sample_submission.csv"
    output = write_submission(args.output, rows, scores, sample)
    print(f"Saved {len(rows)} rows to {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
