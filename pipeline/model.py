"""Shared scoring model, checkpoint format, and prediction helpers."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_ID = "microsoft/deberta-v3-large"
REVISION = "64a8c8eab3e352a784c658aef62be1662607476f"
TRAITS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
CHECKPOINT_SCHEMA = "ell_scoring_checkpoint_v1"


class ScoringModel(nn.Module):
    def __init__(self, backbone, *, has_variance, dropout, max_length):
        super().__init__()
        self.backbone = backbone
        size = backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(size, len(TRAITS))
        self.variance_head = nn.Linear(size, len(TRAITS)) if has_variance else None
        self.max_length = max_length
        if self.variance_head is not None:
            nn.init.zeros_(self.variance_head.weight)
            nn.init.constant_(self.variance_head.bias, math.log(math.expm1(0.24)))

    def features(self, input_ids, attention_mask):
        encoded = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (encoded.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(self, input_ids, attention_mask):
        return self.score_head(self.dropout(self.features(input_ids, attention_mask)))

    def predict_distribution(self, input_ids, attention_mask):
        if self.variance_head is None:
            raise ValueError("This checkpoint has no variance head")
        features = self.features(input_ids, attention_mask)
        return self.score_head(features), 0.01 + nn.functional.softplus(self.variance_head(features))


def new_model(model_path, means, *, teacher, max_length):
    source = str(model_path)
    if Path(source).is_dir():
        asset_args = {"local_files_only": True}
    elif source == MODEL_ID:
        asset_args = {"revision": REVISION}
    else:
        raise ValueError(f"Use {MODEL_ID} or a local DeBERTa model directory")
    config = AutoConfig.from_pretrained(source, **asset_args)
    if config.model_type != "deberta-v2":
        raise ValueError("Expected a DeBERTa-v3 backbone with the deberta-v2 configuration type")
    config._attn_implementation = "eager"
    backbone = AutoModel.from_pretrained(source, config=config, trust_remote_code=False, **asset_args)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **asset_args)
    model = ScoringModel(backbone.float(), has_variance=teacher,
                         dropout=0.0 if teacher else 0.2, max_length=max_length)
    with torch.no_grad():
        nn.init.normal_(model.score_head.weight, mean=0.0, std=0.02)
        model.score_head.bias.copy_(torch.as_tensor(means, dtype=torch.float32))
    return model, tokenizer


def save_scoring_model(directory, model, tokenizer, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(directory / "backbone", safe_serialization=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    heads = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
             if name.startswith(("score_head.", "variance_head."))}
    save_file(heads, directory / "heads.safetensors")
    (directory / "metadata.json").write_text(json.dumps({
        **metadata, "schema": CHECKPOINT_SCHEMA, "traits": list(TRAITS),
        "max_length": model.max_length, "has_variance": model.variance_head is not None,
        "dropout": model.dropout.p,
    }, indent=2) + "\n", encoding="utf-8")


def load_scoring_model(directory, device="cpu"):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != CHECKPOINT_SCHEMA or metadata.get("traits") != list(TRAITS):
        raise ValueError(f"Unsupported scoring checkpoint: {directory}")
    config = AutoConfig.from_pretrained(directory / "backbone", local_files_only=True)
    config._attn_implementation = "eager"
    backbone = AutoModel.from_pretrained(directory / "backbone", config=config,
                                         local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", use_fast=True, local_files_only=True)
    model = ScoringModel(backbone.float(), has_variance=metadata["has_variance"],
                         dropout=metadata["dropout"], max_length=metadata["max_length"])
    weights = load_file(directory / "heads.safetensors")
    expected = {name for name in model.state_dict() if name.startswith(("score_head.", "variance_head."))}
    if set(weights) != expected:
        raise ValueError(f"Checkpoint heads differ from metadata: {directory}")
    for name in ("score_head", "variance_head"):
        head = getattr(model, name)
        if head is not None:
            prefix = f"{name}."
            head.load_state_dict({key[len(prefix):]: value for key, value in weights.items()
                                  if key.startswith(prefix)}, strict=True)
    return model.to(device).eval(), tokenizer


def tokenize_rows(rows, tokenizer, max_length=1024):
    if not 3 <= max_length <= 1024:
        raise ValueError("Maximum input length must be in the range 3 to 1024")
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    if not all(isinstance(value, int) for value in (cls_id, sep_id, pad_id)):
        raise ValueError("The tokenizer must define CLS, SEP, and PAD tokens")
    tokens = []
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        encoded = tokenizer([row["full_text"] for row in chunk], add_special_tokens=False,
                            truncation=True, max_length=max_length - 2,
                            return_attention_mask=False)["input_ids"]
        tokens.extend({**row, "input_ids": [cls_id, *ids, sep_id]} for row, ids in zip(chunk, encoded))
    return tokens


def collate(rows, pad_id):
    length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for index, row in enumerate(rows):
        ids = row["input_ids"]
        input_ids[index, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        mask[index, :len(ids)] = 1
    return input_ids, mask


@torch.inference_mode()
def _predict_tokens(model, rows, pad_id, device, batch_size, *, with_variance=False, features=False):
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    if with_variance and model.variance_head is None:
        raise ValueError("Variance predictions require a teacher checkpoint")
    model.eval()
    order = sorted(range(len(rows)), key=lambda index: len(rows[index]["input_ids"]))
    means = np.empty((len(rows), len(TRAITS)), dtype=np.float32)
    variances = np.empty_like(means) if with_variance else None
    hidden = (np.empty((len(rows), model.backbone.config.hidden_size), dtype=np.float32)
              if features else None)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        args = [tensor.to(device) for tensor in collate([rows[index] for index in indices], pad_id)]
        features_batch = model.features(*args)
        means[indices] = model.score_head(features_batch).cpu().numpy()
        if hidden is not None:
            hidden[indices] = features_batch.cpu().numpy()
        if variances is not None:
            variances[indices] = (0.01 + nn.functional.softplus(model.variance_head(features_batch))).cpu().numpy()
    if not np.isfinite(means).all() or variances is not None and not np.isfinite(variances).all():
        raise FloatingPointError("Model predictions must be finite")
    if features:
        return hidden, means
    return (means, variances) if with_variance else means


def predict_rows(model, tokenizer, rows, device="cpu", batch_size=8, with_variance=False):
    tokens = tokenize_rows(rows, tokenizer, model.max_length)
    return _predict_tokens(model, tokens, tokenizer.pad_token_id, device, batch_size, with_variance=with_variance)
