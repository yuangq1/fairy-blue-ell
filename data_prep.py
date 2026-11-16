"""Prepare ELL role splits and sample earlier Feedback Prize essays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import tempfile
import unicodedata
import zipfile

TRAITS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
SCHEMA = "ell_portable_prepared_v1"
ROLE_NAMES = ("teacher_score_train", "teacher_calibration", "gold_val")
ROLES = frozenset((*ROLE_NAMES, "gold_train", "external"))
HOLDOUT_RECORDS = 2571
HOLDOUT_ZIP_SHA256 = "9ecddfcda83f6a99c8a24dda47fb1c30673fedeaf578a0fadac6645fd151a2f8"
HOLDOUT_IDS_SHA256 = "ec8bf8a2ceb27575c355291f7b3e7c63043341967e9daa940142964abefd2fde"
HOLDOUT_TEXTS_SHA256 = "4f49036519f9b49bb366cb7745792e9968abba674ab1edf6ad86a859b4aadafe"
HOLDOUT_MEMBER = "ELLIPSE_Final_github_test.csv"
SPLIT_SEED = 42
SAMPLE_SEED = 42
SCORES = frozenset(value / 2 for value in range(2, 11))
WORDS = re.compile(r"[^\W_]+", flags=re.UNICODE)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_csv_rows(stream, *, gold: bool) -> dict:
    reader = csv.DictReader(stream)
    columns = set(reader.fieldnames or ())
    id_column = "text_id" if "text_id" in columns else "text_id_kaggle"
    required = {id_column, "full_text", *TRAITS} if gold else {id_column, "full_text"}
    if (gold and id_column != "text_id") or not required.issubset(columns):
        raise ValueError(f"Expected CSV columns {sorted(required)}")
    rows = {}
    for line, row in enumerate(reader, start=2):
        text_id = (row[id_column] or "").strip()
        text = row["full_text"]
        key = text_id.casefold()
        if not text_id or not text or not text.strip() or key in rows:
            raise ValueError(f"Missing text or missing/duplicate ID on CSV row {line}")
        entry = {"text_id": text_id, "full_text": text}
        if gold:
            try:
                values = [float(row[trait]) for trait in TRAITS]
            except (ValueError, TypeError) as error:
                raise ValueError(f"Missing or invalid score on CSV row {line}") from error
            if any(score not in SCORES for score in values):
                raise ValueError(f"Scores must be in 0.5 increments from 1 to 5 on CSV row {line}")
            entry["labels"] = values
        elif "set" in columns and row["set"] != "test":
            raise ValueError("Holdout CSV contains a row outside the official test partition")
        rows[key] = entry
    return rows


def load_holdout(path: Path) -> tuple[dict, dict]:
    if path.suffix.casefold() == ".zip":
        if sha256_file(path) != HOLDOUT_ZIP_SHA256:
            raise ValueError("Expected the pinned official ELLIPSE test archive")
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != [HOLDOUT_MEMBER]:
                raise ValueError("Unexpected official holdout archive contents")
            with archive.open(HOLDOUT_MEMBER, pwd=b"ellipse_test") as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8-sig", newline="") as stream:
                    rows = read_csv_rows(stream, gold=False)
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = read_csv_rows(stream, gold=False)

    pairs = sorted((row["text_id"], sha256_text(normalized(row["full_text"]))) for row in rows.values())
    ids_hash = sha256_text("".join(f"{key}\n" for key, _ in pairs))
    texts_hash = sha256_text("".join(f"{key}\t{digest}\n" for key, digest in pairs))
    if len(pairs) != HOLDOUT_RECORDS or ids_hash != HOLDOUT_IDS_SHA256 or texts_hash != HOLDOUT_TEXTS_SHA256:
        raise ValueError("Holdout membership or text differs from the pinned official ELLIPSE test")
    return rows, {"records": len(rows), "sorted_ids_sha256": ids_hash, "id_text_pairs_sha256": texts_hash}


def fingerprints(text: str) -> tuple[str, str]:
    normalized_text = normalized(text)
    return sha256_text(normalized_text), sha256_text(" ".join(WORDS.findall(normalized_text)))


def document_keys(rows):
    exact, token_text = set(), set()
    for row in rows:
        text_key, word_key = fingerprints(row["full_text"])
        exact.add(text_key)
        token_text.add(word_key)
    return exact, token_text


def split_gold(gold: dict, ids_file: Path | None) -> tuple[dict, dict]:
    if len(gold) < 3:
        raise ValueError("At least three gold essays are needed for separate training, calibration and validation")
    n_eval = len(gold) // 10 if len(gold) >= 10 else 1
    counts = (len(gold) - 2 * n_eval, n_eval, n_eval)
    if ids_file is None:
        ids = sorted(gold)
        random.Random(SPLIT_SEED).shuffle(ids)
        chunks = [ids[:counts[0]], ids[counts[0]:-n_eval], ids[-n_eval:]]
        split = {"kind": "deterministic", "seed": SPLIT_SEED}
    else:
        payload = json.loads(ids_file.read_text(encoding="utf-8-sig"))
        roles = payload.get("roles", payload)
        chunks = []
        for name in ROLE_NAMES:
            ids = roles[name]
            if isinstance(ids, dict):
                ids = ids["document_ids"]
            if not isinstance(ids, list) or any(not isinstance(key, str) for key in ids):
                raise ValueError(f"Expected a list of document IDs for {name}")
            chunks.append([key.strip().casefold() for key in ids])
        split = {"kind": "provided_ordered_ids", "sha256": sha256_file(ids_file)}
    flat = [key for chunk in chunks for key in chunk]
    if tuple(map(len, chunks)) != counts or len(flat) != len(set(flat)) or set(flat) != set(gold):
        raise ValueError("Split IDs must partition the gold data in an 80/10/10 split")
    roles = {name: [gold[key] for key in chunk] for name, chunk in zip(ROLE_NAMES, chunks)}
    roles["gold_train"] = roles["teacher_score_train"] + roles["teacher_calibration"]
    return roles, split


def sample_external(earlier_dirs: list[Path], forbidden_ids: set, forbidden_exact: set,
                    forbidden_tokens: set, sample_size: int) -> tuple[list, int]:
    pool = {}
    seen_exact = set(forbidden_exact)
    seen_tokens = set(forbidden_tokens)
    for root in earlier_dirs:
        if not root.is_dir():
            raise ValueError(f"Expected an extracted earlier Feedback Prize train directory: {root}")
        paths = sorted(root.rglob("*.txt"), key=lambda path: (path.relative_to(root).as_posix().casefold(),
                                                              path.relative_to(root).as_posix()))
        if not paths:
            raise ValueError(f"No .txt essays found in {root}")
        for path in paths:
            text_id = path.stem
            key = text_id.casefold()
            if key in forbidden_ids or key in pool:
                continue
            text = path.read_text(encoding="utf-8-sig")
            if not text.strip():
                continue
            exact, word_key = fingerprints(text)
            if exact in seen_exact or word_key in seen_tokens:
                continue
            seen_exact.add(exact)
            seen_tokens.add(word_key)
            pool[key] = {"text_id": text_id, "full_text": text}
    if sample_size < 1 or len(pool) < sample_size:
        raise ValueError(f"Requested {sample_size} external essays from {len(pool)} eligible files")
    keys = random.Random(SAMPLE_SEED).sample(sorted(pool), sample_size)
    return [pool[key] for key in keys], len(pool)


def read_prepared_manifest(data_dir) -> dict:
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != SCHEMA or set(manifest.get("files", {})) != ROLES:
        raise ValueError("Expected a prepared ELL data directory")
    if tuple(manifest.get("traits", ())) != TRAITS:
        raise ValueError("Prepared trait order differs")
    if manifest.get("holdout") != {"records": HOLDOUT_RECORDS, "sorted_ids_sha256": HOLDOUT_IDS_SHA256,
                                   "id_text_pairs_sha256": HOLDOUT_TEXTS_SHA256}:
        raise ValueError("Missing verified ELLIPSE holdout identity")
    return manifest


def load_prepared_role(data_dir, role: str) -> list:
    if role not in ROLES:
        raise ValueError(f"Unknown prepared role: {role}")
    receipt = read_prepared_manifest(data_dir)["files"][role]
    if receipt["path"] != f"{role}.jsonl":
        raise ValueError(f"Unexpected prepared filename for {role}")
    path = Path(data_dir) / receipt["path"]
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError(f"Prepared role file changed: {role}")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != receipt["records"]:
        raise ValueError(f"Prepared role count differs: {role}")
    return rows


def prepare(*, ell_train: Path, earlier_dirs: list[Path], holdout_inputs: Path, output_dir: Path,
            split_ids: Path | None = None, sample_size: int = 4000) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Prepared output already exists: {output_dir}")
    holdout, holdout_receipt = load_holdout(holdout_inputs)
    with ell_train.open(encoding="utf-8-sig", newline="") as stream:
        gold = read_csv_rows(stream, gold=True)
    holdout_exact, holdout_tokens = document_keys(holdout.values())
    gold_exact, gold_tokens = document_keys(gold.values())
    if gold.keys() & holdout.keys() or gold_exact & holdout_exact or gold_tokens & holdout_tokens:
        raise ValueError("Gold data overlaps the fixed holdout")
    roles, split = split_gold(gold, split_ids)
    if earlier_dirs:
        roles["external"], pool_count = sample_external(
            earlier_dirs, set(gold) | set(holdout), gold_exact | holdout_exact,
            gold_tokens | holdout_tokens, sample_size
        )
    else:
        roles["external"], pool_count = [], 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ell_prepare_", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        files = {}
        for role, rows in roles.items():
            filename = f"{role}.jsonl"
            path = stage / filename
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            files[role] = {"path": filename, "records": len(rows), "sha256": sha256_file(path)}
        manifest = {"schema": SCHEMA, "traits": list(TRAITS), "holdout": holdout_receipt,
                    "gold_source_sha256": sha256_file(ell_train), "split": split,
                    "external_sampling": {"seed": SAMPLE_SEED, "eligible_pool_size": pool_count,
                                          "sample_size": len(roles["external"]), "input_directory_count": len(earlier_dirs)},
                    "files": files}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ell-train", required=True, type=Path)
    parser.add_argument("--earlier-dir", default=[], type=Path, action="append",
                        help="Extracted train directory containing .txt essays. Repeat for additional corpora.")
    parser.add_argument("--holdout-inputs", required=True, type=Path,
                        help="Pinned official ELLIPSE test ZIP or its extracted CSV")
    parser.add_argument("--split-ids", type=Path, help="Optional saved role_id_lists.json for the existing split")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=4000)
    args = parser.parse_args()
    manifest = prepare(ell_train=args.ell_train, earlier_dirs=args.earlier_dir,
                       holdout_inputs=args.holdout_inputs, output_dir=args.output_dir,
                       split_ids=args.split_ids, sample_size=args.sample_size)
    print(json.dumps({name: item["records"] for name, item in manifest["files"].items()}))


if __name__ == "__main__":
    main()
