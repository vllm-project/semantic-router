#!/usr/bin/env python3
"""Train/test split with sha256 manifest freeze for evaluation discipline.

Given a JSONL dataset, performs a stratified (or random) configurable-ratio
split, writes train and eval files, and freezes a sha256 manifest so
subsequent runs can detect if the split changed (preventing train/test
contamination across configuration iterations).

If a manifest already exists at the target path, the script verifies that
the input hash, seed, ratio, and stratify key match before writing. A
mismatch is rejected unless --force is passed, so a changed seed or input
cannot silently move held-out examples into calibration.

Usage:
    python split_evalset.py \\
        --input tasks/accept_office.jsonl \\
        --output-dir tasks/ \\
        --seed 42 \\
        [--ratio 0.5] \\
        [--stratify-key source] \\
        --manifest tasks/split_manifest.json
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stratified_split(
    records: List[dict], key: str, ratio: float, seed: int
) -> Tuple[List, List]:
    """Split records preserving stratum proportions."""
    by_stratum = defaultdict(list)
    for r in records:
        by_stratum[r.get(key, "default")].append(r)

    train, eval_ = [], []
    rng = random.Random(seed)
    for stratum, items in by_stratum.items():
        rng.shuffle(items)
        n = len(items)
        split_idx = int(n * ratio)
        train.extend(items[:split_idx])
        eval_.extend(items[split_idx:])
    return train, eval_


def random_split(records: List[dict], ratio: float, seed: int) -> Tuple[List, List]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    split_idx = int(n * ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    parser = argparse.ArgumentParser(
        description="Train/test split with sha256 manifest freeze"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Train ratio (default 0.5 = half/half)",
    )
    parser.add_argument(
        "--stratify-key",
        default=None,
        help="Field name for stratified split (e.g., 'source')",
    )
    parser.add_argument("--prefix", default="evalset", help="Output file prefix")
    parser.add_argument(
        "--manifest", default=None, help="Manifest JSON path (stores sha256 for freeze)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing manifest even if seed/ratio/input changed",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    if args.stratify_key:
        train, eval_ = stratified_split(
            records, args.stratify_key, args.ratio, args.seed
        )
        print(
            f"Stratified split by '{args.stratify_key}': "
            f"train={len(train)}, eval={len(eval_)}"
        )
    else:
        train, eval_ = random_split(records, args.ratio, args.seed)
        print(f"Random split: train={len(train)}, eval={len(eval_)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / f"{args.prefix}_train.jsonl"
    eval_path = out_dir / f"{args.prefix}_eval.jsonl"

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = out_dir / f"{args.prefix}_manifest.json"

    # Freeze check: if a manifest already exists, verify that the input
    # hash and split parameters match before allowing an overwrite. This
    # prevents a second invocation with a different seed from silently
    # replacing the held-out eval set and printing "FROZEN" anyway.
    input_hash = sha256_file(args.input)
    if manifest_path.exists() and not args.force:
        try:
            prev = json.loads(manifest_path.read_text())
        except Exception:
            prev = {}
        prev_input = prev.get("sha256_input")
        prev_seed = prev.get("seed")
        prev_ratio = prev.get("ratio")
        prev_stratify = prev.get("stratify_key")
        mismatches = []
        if prev_input is not None and prev_input != input_hash:
            mismatches.append("input file changed")
        if prev_seed is not None and prev_seed != args.seed:
            mismatches.append(f"seed {prev_seed} -> {args.seed}")
        if prev_ratio is not None and prev_ratio != args.ratio:
            mismatches.append(f"ratio {prev_ratio} -> {args.ratio}")
        if prev_stratify is not None and prev_stratify != args.stratify_key:
            mismatches.append(f"stratify_key {prev_stratify} -> {args.stratify_key}")
        if mismatches:
            print(
                "ERROR: manifest already exists and split parameters differ: "
                + "; ".join(mismatches)
            )
            print(
                "Pass --force to overwrite and re-freeze a new split. "
                "Without --force, the existing eval set is preserved."
            )
            sys.exit(1)
        else:
            print(
                f"Manifest {manifest_path} exists and split parameters match; "
                f"re-writing split artifacts (content-identical)."
            )

    with open(train_path, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(eval_path, "w") as f:
        for r in eval_:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "seed": args.seed,
        "ratio": args.ratio,
        "stratify_key": args.stratify_key,
        "n_train": len(train),
        "n_eval": len(eval_),
        "n_total": len(records),
        "sha256_input": input_hash,
        "sha256_train": sha256_file(str(train_path)),
        "sha256_eval": sha256_file(str(eval_path)),
        "discipline": "eval set is FROZEN: use train for calibration/tuning; "
        "eval runs once at the final experiment to prevent overfitting",
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Train: {train_path} (sha256: {manifest['sha256_train'][:16]}...)")
    print(f"Eval:  {eval_path} (sha256: {manifest['sha256_eval'][:16]}...)")
    print(f"Manifest: {manifest_path}")
    print(f"Discipline: {manifest['discipline']}")


if __name__ == "__main__":
    main()
