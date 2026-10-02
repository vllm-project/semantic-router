"""Milestone 7 data helpers for the ~27B track (host CPU, stdlib only).

``ml-upsample``: the ML block (decoder M15's multilingual-preserving upsample, applied to a 27B mixture). The IB blocks
are mostly English, so a mixture that adds them carries a lower multilingual share than the released a20 TRAIN. The
output is the mixture's lines byte for byte and in order, then one copy (id suffix ``~m2``) of chosen whole a20 groups
whose rows are all non-``en``, stratified by language in proportion to a20's multilingual mass, adding
U = (s * T - ML) / (1 - s) so the output's multilingual share equals a20's share s. Mass is the UTF-8 length of each
row's model-visible fields (instructions, state, options), a deterministic token proxy. Groups are taken whole in a
seeded order per language while they fit the language's quota; the build fails if the share is off s by more than TOL.

    python3 -m v2.27b.m7.m7_data ml-upsample --base a20.train.jsonl --input MIX.train.jsonl --seed S --output OUT.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ML_SUFFIX = "~m2"
TOL = 0.002


def mass(row: dict[str, Any]) -> int:
    fields = (row.get("instructions"), row.get("state"), row.get("options"))
    return sum(
        len(
            (
                f
                if isinstance(f, str)
                else json.dumps(f, ensure_ascii=False, sort_keys=True)
            ).encode("utf-8")
        )
        for f in fields
        if f is not None
    )


def share(rows: list[dict[str, Any]]) -> tuple[int, int]:
    total = ml = 0
    for r in rows:
        m = mass(r)
        total += m
        if r["language"] != "en":
            ml += m
    return ml, total


def ml_groups(
    base: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, list[int]]], Counter]:
    """a20 groups whose rows are all non-en, by their main language, and multilingual mass per language."""
    members: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(base):
        members[r["group_id"]].append(i)
    groups: dict[str, dict[str, list[int]]] = defaultdict(dict)
    lang_mass: Counter = Counter()
    for g, idx in members.items():
        langs: Counter = Counter()
        for i in idx:
            langs[base[i]["language"]] += mass(base[i])
        if "en" in langs:
            continue
        lang_mass.update(langs)
        groups[langs.most_common(1)[0][0]][g] = idx
    return groups, lang_mass


def upsample(base: list[dict[str, Any]], target: int, seed: str) -> list[int]:
    groups, lang_mass = ml_groups(base)
    ml = sum(lang_mass.values())
    chosen: list[int] = []
    for lang in sorted(groups):
        quota = target * lang_mass[lang] / ml
        order = sorted(groups[lang])
        random.Random(f"{seed}:{lang}").shuffle(order)
        taken = 0
        for g in order:
            size = sum(mass(base[i]) for i in groups[lang][g])
            if taken + size > quota:
                continue
            taken += size
            chosen.extend(groups[lang][g])
    return chosen


def ml_upsample(
    base_path: Path, source: Path, seed: str, output: Path
) -> dict[str, Any]:
    data = source.read_bytes()
    base_data = base_path.read_bytes()
    lines = data.splitlines(keepends=True)
    if any(not line.strip() for line in lines):
        raise ValueError(f"{source}: blank line")
    rows = [json.loads(line) for line in lines]
    base = [json.loads(line) for line in base_data.splitlines()]
    if not set(base_data.splitlines(keepends=True)) <= set(lines):
        raise ValueError(f"{source} misses a20 rows")
    ids = {r["id"] for r in rows}
    if any(i.endswith(ML_SUFFIX) for i in ids):
        raise ValueError(f"{source} already has ML copies")
    s_ml, s_total = share(base)
    s = s_ml / s_total
    ml, total = share(rows)
    target = max(0, round((s * total - ml) / (1 - s)))
    chosen = upsample(base, target, seed)
    copies = []
    for i in chosen:
        row = dict(base[i])
        row["id"] = row["id"] + ML_SUFFIX
        if row["id"] in ids:
            raise ValueError(f"copy id {row['id']} exists")
        copies.append((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    out = data + (b"" if data.endswith(b"\n") or not data else b"\n") + b"".join(copies)
    added = [base[i] for i in chosen]
    a_ml, a_total = share(added)
    built = (ml + a_ml) / (total + a_total)
    if abs(built - s) > TOL:
        raise ValueError(
            f"built multilingual share {built:.4f} is off a20's {s:.4f} by more than {TOL}"
        )
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(out)
    return {
        "schema": "m7-ml-upsample/1",
        "base": str(base_path),
        "base_sha256": hashlib.sha256(base_data).hexdigest(),
        "input": str(source),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "seed": seed,
        "a20_ml_share": round(s, 6),
        "input_ml_share": round(ml / total, 6),
        "target_mass": target,
        "added_mass": a_total,
        "added_rows": len(chosen),
        "added_groups": len({base[i]["group_id"] for i in chosen}),
        "added_rows_by_language": dict(
            sorted(Counter(r["language"] for r in added).items())
        ),
        "output_ml_share": round(built, 6),
        "rows_out": len(rows) + len(chosen),
        "output": str(output),
        "output_sha256": hashlib.sha256(out).hexdigest(),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ml-upsample")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    print(
        json.dumps(
            ml_upsample(args.base, args.input, args.seed, args.output),
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
