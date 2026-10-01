"""Decoder M11 stage-2 TRAIN files (prereg dec-m11-stage2-prereg-2026-10-01.md, "Ratio and matched tokens").

Counts every row's tokens with the trainer's head-readout encoding (`training.model.decision_model.encode`, the
4B base tokenizer, max length 8,192), so T = LH's TRAIN tokens as the trainer counts them. The IB token target is
B = min(round(share * T), tokens of the transfer-only pool), the same for both arms. Both arms share one base
subsample of about T - B tokens; 4b-LHB fills B from every IB family, 4b-LHBx from the families other than the
in-distribution ones. Sampling is by whole groups, stratified by family in proportion to the family's tokens in its
pool, groups in a seeded shuffle per family, taken while the family stays within its quota. Lines are copied byte for
byte (base, then IB1, then IB2, each in file order).

usage: m11_s2data.py --base B --base-sha S --ib1 F --ib1-sha S --ib2 F --ib2-sha S --tokenizer DIR \
         --indist w2c,isarc,hover,gsm2 --output DIR [--share 0.25] [--seed 20261001] [--workers N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

SCHEMA = "dec-m11-s2data/1"
MAX_LENGTH = 8192
_TOKENIZER: Any = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _init(tokenizer_dir: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)


def _length(line: bytes) -> int:
    from training.model.decision_model import encode

    return len(encode(json.loads(line), _TOKENIZER, 1 << 30)["ids"])


def sample_groups(
    rows: list[dict[str, Any]], target: int, seed: int
) -> tuple[set[int], dict[str, Any]]:
    """Indices of rows kept: whole groups, stratified by family, about `target` tokens in total."""
    total = sum(r["tokens"] for r in rows)
    if target >= total:
        return set(range(len(rows))), {"fraction": 1.0}
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    fam_tokens: dict[str, int] = defaultdict(int)
    for i, r in enumerate(rows):
        groups[r["family"]][r["group_id"]].append(i)
        fam_tokens[r["family"]] += r["tokens"]
    kept: set[int] = set()
    for family in sorted(groups):
        quota = target * fam_tokens[family] / total
        rng = random.Random(f"{seed}:{family}")
        order = sorted(groups[family])
        rng.shuffle(order)
        used = 0
        for g in order:
            size = sum(rows[i]["tokens"] for i in groups[family][g])
            if used + size > quota:
                continue
            kept.update(groups[family][g])
            used += size
    return kept, {"fraction": target / total}


def summary(rows: list[dict[str, Any]], kept: set[int]) -> dict[str, Any]:
    fam: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "tokens": 0})
    for i in sorted(kept):
        fam[rows[i]["family"]]["rows"] += 1
        fam[rows[i]["family"]]["tokens"] += rows[i]["tokens"]
    return {
        "rows": len(kept),
        "tokens": sum(rows[i]["tokens"] for i in kept),
        "groups": len({rows[i]["group_id"] for i in kept}),
        "families": dict(sorted(fam.items())),
    }


def load(path: Path, block: str, lengths: list[int]) -> list[dict[str, Any]]:
    rows = []
    with path.open("rb") as stream:
        for line, n in zip(stream, lengths, strict=True):
            row = json.loads(line)
            rows.append(
                {
                    "block": block,
                    "family": row["family"],
                    "group_id": row["group_id"],
                    "id": row["id"],
                    "tokens": n,
                    "line": line if line.endswith(b"\n") else line + b"\n",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "ib1", "ib2"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--indist", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--share", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args(argv)
    indist = set(a.indist.split(","))
    files = {"base": a.base, "ib1": a.ib1, "ib2": a.ib2}
    for name, path in files.items():
        if sha256(path) != getattr(a, f"{name}_sha"):
            raise ValueError(f"{path}: sha256 differs from --{name}-sha")
    blocks: dict[str, list[dict[str, Any]]] = {}
    with Pool(a.workers, initializer=_init, initargs=(a.tokenizer,)) as pool:
        for name, path in files.items():
            with path.open("rb") as stream:
                lengths = pool.map(_length, list(stream), chunksize=64)
            blocks[name] = load(path, name, lengths)
    longest = max(r["tokens"] for rows in blocks.values() for r in rows)
    if longest > MAX_LENGTH:
        raise ValueError(f"a row has {longest} tokens > {MAX_LENGTH}")
    ids = [r["id"] for rows in blocks.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("repeated row id across the inputs")
    base = blocks["base"]
    ib = blocks["ib1"] + blocks["ib2"]
    present = {r["family"] for r in ib}
    if not indist <= present:
        raise ValueError(
            f"in-distribution families not in IB: {sorted(indist - present)}"
        )
    total = sum(r["tokens"] for r in base)
    pool_b = [r for r in ib if r["family"] not in indist]
    target = min(round(a.share * total), sum(r["tokens"] for r in pool_b))
    base_kept, base_info = sample_groups(base, total - target, a.seed)
    arms = {"4b-LHB": ib, "4b-LHBx": pool_b}
    a.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "inputs": {
            n: {"path": str(p), "sha256": getattr(a, f"{n}_sha")}
            for n, p in files.items()
        },
        "tokenizer": a.tokenizer,
        "token_unit": "training.model.decision_model.encode (head readout), no truncation",
        "share": a.share,
        "seed": a.seed,
        "in_distribution": sorted(indist),
        "T_base_tokens": total,
        "base_rows": len(base),
        "ib_pool_tokens": {
            "all": sum(r["tokens"] for r in ib),
            "transfer": sum(r["tokens"] for r in pool_b),
        },
        "B_target": target,
        "longest_row": longest,
        "base_subsample": {**summary(base, base_kept), **base_info},
        "arms": {},
    }
    for arm, pool_rows in arms.items():
        kept, info = sample_groups(pool_rows, target, a.seed)
        out = a.output / arm
        out.mkdir()
        chosen = [base[i] for i in sorted(base_kept)] + [
            pool_rows[i] for i in sorted(kept)
        ]
        with (out / "train.jsonl").open("xb") as sink:
            for r in chosen:
                sink.write(r["line"])
        with (out / "train.ids.jsonl").open("x") as sink:
            for r in chosen:
                sink.write(
                    json.dumps(
                        {"id": r["id"], "block": r["block"], "tokens": r["tokens"]}
                    )
                    + "\n"
                )
        ib_sum = summary(pool_rows, kept)
        tokens = sum(r["tokens"] for r in chosen)
        report["arms"][arm] = {
            "train_sha256": sha256(out / "train.jsonl"),
            "ids_sha256": sha256(out / "train.ids.jsonl"),
            "rows": len(chosen),
            "tokens": tokens,
            "tokens_vs_T": tokens / total,
            "ib": {**ib_sum, **info, "share_of_tokens": ib_sum["tokens"] / tokens},
        }
    (a.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                "T": total,
                "B": target,
                "arms": {
                    k: {
                        "rows": v["rows"],
                        "tokens": v["tokens"],
                        "ib_share": round(v["ib"]["share_of_tokens"], 4),
                    }
                    for k, v in report["arms"].items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
