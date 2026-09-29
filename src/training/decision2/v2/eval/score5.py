"""Score5-DEV v1: a 5-level Score development check (never a release score).

    python3 -m v2.eval.score5 build --source <A7q aho.jsonl> --source-sha256 <sha> --output-dir <run dir>

`build` draws 100 rows per gold level from the A7q held-out slice (OASST1 reply ratings,
four axes, five levels), spread over the axes and at most one row per source group,
and writes gold-free prompts, HT-DEV-format gold lines, the selected source rows
(`panel-rows.jsonl`, which fitting code must exclude; every other row is the fit pool)
and a MANIFEST. Items keep the row's state and instructions and list the five level
descriptions in ascending order as Score criteria. Rules:
`v2/eval/records/score5-dev-prereg-2026-09-29.md`.

`summary` (used by `v2.eval.dev_readout`) gives the level histogram, modal share, rare
levels, invalid/missing count, accuracy with a 95% item-bootstrap CI, macro-F1, QWK and
the COLLAPSE / WARN / NO-SIGNAL flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2.eval.htdev.build import code_commit, jsonl_bytes, private_dir, sha_bytes
from v2.eval.sealed.build import gold_record, write_new
from v2.eval.sealed.schema import LONG_INPUT_CHARS, MAX_INPUT_CHARS, input_chars
from v2.eval.sealed.score import macro_f1, outcomes, quadratic_kappa

SCHEMA = "dev2-score5-build/1"
PANEL = "score5-dev"
SOURCE = "a7q-aho"
SOURCE_FILE = "v2/a7/arms/A7q/aho.jsonl"
SEED = 20260929
LEVELS = 5
PER_LEVEL = 100
GROUP_CAPS = (1, 2)
REPLICATES = 2000
RARE_SHARE = 0.02
COLLAPSE_MODAL = 0.60
WARN_MODAL = 0.40
CHANCE = 0.20


def axis(row: dict[str, Any]) -> str:
    return row["audit_metadata"]["a7"]["axis"]


def order_key(row_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{row_id}".encode()).hexdigest()


def panel_id(row_id: str) -> str:
    return "score5-" + hashlib.sha256(f"{SEED}:id:{row_id}".encode()).hexdigest()[:16]


def question(row: dict[str, Any]) -> dict[str, Any]:
    options = sorted(row["options"], key=lambda o: int(o["key"]))
    if [o["key"] for o in options] != [str(i) for i in range(LEVELS)]:
        raise ValueError(f"{row['id']}: options are not levels 0..{LEVELS - 1}")
    return {
        "type": "score",
        "instructions": row["instructions"],
        "criteria": [o["description"] for o in options],
    }


def select(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """PER_LEVEL rows per level: level x axis cells filled round-robin (scarcest cell
    first in each round, seeded row order, unused groups only), then water-filled onto
    the level's other axes; a group cap of two only if one row per group cannot fill."""
    axes = sorted({axis(r) for r in rows})
    cells: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: order_key(r["id"])):
        cells[(row["label"], axis(row))].append(row)
    target = PER_LEVEL // len(axes)
    used: Counter = Counter()
    taken: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    chosen_ids: set[str] = set()

    def next_row(cell, cap):
        for row in cells[cell]:
            if row["id"] not in chosen_ids and used[row["group_id"]] < cap:
                return row
        return None

    def take(cell, row):
        taken[cell].append(row)
        chosen_ids.add(row["id"])
        used[row["group_id"]] += 1

    def level_total(level: int) -> int:
        return sum(len(taken[(level, a)]) for a in axes)

    cap_used = GROUP_CAPS[0]
    for cap in GROUP_CAPS:
        cap_used = cap
        order = sorted(cells, key=lambda c: (len(cells[c]), c))
        progress = True
        while progress:
            progress = False
            for cell in order:
                if len(taken[cell]) < target and level_total(cell[0]) < PER_LEVEL:
                    if row := next_row(cell, cap):
                        take(cell, row)
                        progress = True
        for level in range(LEVELS):
            while level_total(level) < PER_LEVEL:
                open_axes = sorted(
                    (len(taken[(level, a)]), a)
                    for a in axes
                    if next_row((level, a), cap) is not None
                )
                if not open_axes:
                    break
                cell = (level, open_axes[0][1])
                take(cell, next_row(cell, cap))
        if len(chosen_ids) == PER_LEVEL * LEVELS:
            break
    chosen = [row for cell in sorted(taken) for row in taken[cell]]
    if len(chosen) != PER_LEVEL * LEVELS:
        raise ValueError(f"only {len(chosen)} rows could be selected")
    info = {
        "group_cap": cap_used,
        "max_rows_per_group": max(used.values()),
        "groups": len(used),
        "cells": {f"{lv}|{a}": len(taken[(lv, a)]) for lv, a in sorted(taken)},
        "eligible_cells": {f"{lv}|{a}": len(cells[(lv, a)]) for lv, a in sorted(cells)},
    }
    return chosen, info


def render(row: dict[str, Any], index: int) -> tuple[dict, dict]:
    item_id = panel_id(row["id"])
    q = question(row)
    chars = input_chars(row["state"], q)
    if chars > MAX_INPUT_CHARS:
        raise ValueError(f"{row['id']}: input over {MAX_INPUT_CHARS} characters")
    prompt = {"id": item_id, "state": row["state"], "questions": {"decision": q}}
    gold = {
        "id": item_id,
        "task": f"score5/{axis(row)}",
        "source": SOURCE,
        "split": row["split"],
        "source_item_id": row["id"],
        "group_id": row["group_id"],
        "cluster_id": row["group_id"],
        "language": row["language"],
        "input_chars": chars,
        "long": chars >= LONG_INPUT_CHARS,
        "provenance": {"file": SOURCE_FILE, "row": index},
        "questions": {"decision": q},
        "gold": {"decision": gold_record({"question": q, "gold": row["label"]})},
    }
    return prompt, gold


def build(args: argparse.Namespace) -> int:
    source_bytes = args.source.read_bytes()
    source_sha = sha_bytes(source_bytes)
    if source_sha != args.source_sha256:
        raise SystemExit(f"source sha256 {source_sha} != expected {args.source_sha256}")
    rows = [json.loads(line) for line in source_bytes.decode().splitlines() if line]
    index = {row["id"]: i for i, row in enumerate(rows)}
    if len(index) != len(rows):
        raise ValueError("duplicate source row ids")
    chosen, info = select(rows)
    pairs = sorted(
        (render(row, index[row["id"]]) for row in chosen), key=lambda p: p[0]["id"]
    )
    if len({p["id"] for p, _ in pairs}) != len(pairs):
        raise ValueError("panel id collision")
    panel_rows = sorted(
        (
            {
                "panel_id": panel_id(row["id"]),
                "source_row_id": row["id"],
                "input_sha256": row["input_sha256"],
                "group_id": row["group_id"],
            }
            for row in chosen
        ),
        key=lambda r: r["source_row_id"],
    )
    private_dir(args.output_dir)
    files = {
        f"{PANEL}.prompts.jsonl": jsonl_bytes([p for p, _ in pairs]),
        f"{PANEL}.gold.jsonl": jsonl_bytes([g for _, g in pairs]),
        "panel-rows.jsonl": jsonl_bytes(panel_rows),
    }
    for name, data in files.items():
        write_new(args.output_dir / name, data)

    def counts(key) -> dict[Any, int]:
        return dict(sorted(Counter(map(key, chosen)).items()))

    manifest = {
        "schema": SCHEMA,
        "panel": PANEL,
        "scope": "development readout only; never a release score, v3, chart or card",
        "familiarity": "FAMILIAR: held out by group hash, but A7q training groups overlap by n-grams",
        "seed": SEED,
        "code_commit": code_commit(),
        "source": {"file": SOURCE_FILE, "sha256": source_sha, "rows": len(rows)},
        "items": len(chosen),
        "fit_pool_rows": len(rows) - len(chosen),
        "fit_pool_rule": "every source row whose id is not in panel-rows.jsonl",
        "by_level": counts(lambda r: r["label"]),
        "by_axis": counts(axis),
        "by_level_axis": info["cells"],
        "eligible_by_level_axis": info["eligible_cells"],
        "by_language": counts(lambda r: r["language"]),
        "groups": info["groups"],
        "group_cap": info["group_cap"],
        "max_rows_per_group": info["max_rows_per_group"],
        "long_inputs": sum(g["long"] for _, g in pairs),
        "selected_row_id_sha256": sorted(
            hashlib.sha256(r["id"].encode()).hexdigest() for r in chosen
        ),
        "files_sha256": {name: sha_bytes(data) for name, data in files.items()},
    }
    data = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    write_new(args.output_dir / "MANIFEST.json", data)
    print(
        json.dumps(
            {
                "items": len(chosen),
                "fit_pool_rows": manifest["fit_pool_rows"],
                "group_cap": info["group_cap"],
                "manifest_sha256": sha_bytes(data),
                **manifest["files_sha256"],
            }
        )
    )
    return 0


def flags(modal_share: float | None, rare: int, ci_low: float) -> list[str]:
    out = []
    if modal_share is None or modal_share >= COLLAPSE_MODAL or rare >= 2:
        out.append("COLLAPSE")
    elif modal_share >= WARN_MODAL or rare == 1:
        out.append("WARN")
    if ci_low <= CHANCE:
        out.append("NO-SIGNAL")
    return out


def level_usage(predicted: list[int | None]) -> dict[str, Any]:
    answered = [p for p in predicted if p is not None]
    histogram = {str(level): answered.count(level) for level in range(LEVELS)}
    top = max(histogram.values())
    return {
        "histogram": histogram,
        "answered": len(answered),
        "invalid_or_missing": len(predicted) - len(answered),
        "modal_level": (
            min(int(k) for k, v in histogram.items() if v == top) if answered else None
        ),
        "modal_share": top / len(answered) if answered else None,
        "rare_levels": sorted(
            int(k) for k, v in histogram.items() if v < RARE_SHARE * len(answered)
        ),
    }


def bootstrap_ci(correct: list[bool], replicates: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(correct)
    draws = sorted(
        sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates)
    )
    return [draws[int(0.025 * replicates)], draws[int(0.975 * replicates) - 1]]


def summary(
    gold: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    replicates: int = REPLICATES,
) -> dict[str, Any]:
    unknown = set(predictions) - {row["id"] for row in gold}
    if unknown:
        raise ValueError(f"{len(unknown)} prediction ids are not in {PANEL}")
    rows = outcomes(gold, predictions)
    pairs = [(r[5], r[6]) for r in rows]
    correct = [g == p for g, p in pairs]
    usage = level_usage([p for _, p in pairs])
    ci = bootstrap_ci(correct, replicates, SEED)
    gold_counts = Counter(g for g, _ in pairs)
    return {
        "scope": "development check only (FAMILIAR in-family panel); never a release score",
        "n": len(rows),
        **usage,
        "rare_level_count": len(usage["rare_levels"]),
        "correct": sum(correct),
        "accuracy": sum(correct) / len(rows),
        "accuracy_ci95": ci,
        "bootstrap": {"replicates": replicates, "seed": SEED},
        "macro_f1": macro_f1(pairs),
        "qwk_answered": quadratic_kappa(pairs),
        "always_modal_accuracy": max(gold_counts.values()) / len(rows),
        "by_axis": {
            task: sum(r[5] == r[6] for r in rows if r[0] == task)
            / sum(r[0] == task for r in rows)
            for task in sorted({r[0] for r in rows})
        },
        "flags": flags(usage["modal_share"], len(usage["rare_levels"]), ci[0]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("build")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--source-sha256", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
