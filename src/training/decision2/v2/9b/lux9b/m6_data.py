"""Milestone 6 CPU data builds on the K recipe (x60) for the 9B track.

* ``split``: the soft-target set S (the human-rated rows of x60, by a frozen pool rule), the
  S rows that already have AutoJev-27B targets in the production waves, and the inputs of the
  one new AutoJev wave (canonical rows + native prompts) for the rest. S rows whose native
  prompt is longer than the longest prompt of the production waves stay out of S (they keep
  own-Lux targets), so the new wave stays inside the prompt range the frozen autotune cache
  has already served.
* ``ka``: the KA teacher file: AutoJev-27B targets on S (production waves + the new wave),
  the x60 own-Lux targets on every other row; train.jsonl is x60, byte for byte.
* ``kh``: the KH TRAIN: x60 cut to (x60 tokens - block tokens) in whole groups, stratified
  by pool x source x task type x language as the x60 recipe was, plus the HS1 block (every
  ``hs1_unmet_condition`` row and whole ``hs1_quote_check`` groups up to half of its native
  tokens in ``sha256("<seed>:f1:" + group_id)`` order). Kept x60 rows keep their own-Lux
  targets; block rows have none (``--teacher-partial``: gold only).

    python3 -m lux9b.m6_data split --spec SPEC --root NAME=PATH ... --output-dir OUT
    python3 -m lux9b.m6_data ka --spec SPEC --root ... --wave-targets T --wave-report R --output-dir OUT
    python3 -m lux9b.m6_data kh --spec SPEC --root ... --tokenizer /model --output-dir OUT

Every input is hash-verified; outputs go to a new directory with ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lux9b.m3_data import (
    c1_keys,
    check_teacher,
    denied_hits,
    read_jsonl,
    recipe_budget,
    resolve,
    take_prefix,
    token_lengths,
    verified,
    write_lines,
)
from training.model.data import (
    canonical,
    check_partition_isolation,
    file_sha256,
    load_partition,
)

SCHEMA = "decision2-9b-m6-data/1"
F1 = "hs1_quote_check"
F3 = "hs1_unmet_condition"


def human_rated(row: dict[str, Any], pool: str, spec: dict[str, Any]) -> bool:
    """The frozen S rule: a row is human-rated if its pool lists its source (or "*") under
    ``human_sources``; replay rows (``replay_source``) count only if they carry their upstream
    human label. Every pool of the recipe must be named in ``pools``."""
    rule = spec["soft_target_rule"]
    if pool not in rule["pools"]:
        raise ValueError(f"pool {pool} is not named by the soft-target rule")
    sources = rule["human_sources"].get(pool, [])
    if row["source"] == rule["replay_source"]:
        return row["source"] in sources and "upstream_label" in row
    return "*" in sources or row["source"] in sources


def load_x60(spec, roots, inputs):
    rows = read_jsonl(verified(spec["x60"]["train"], roots, inputs))
    teacher = {
        r["id"]: r for r in read_jsonl(verified(spec["x60"]["teacher"], roots, inputs))
    }
    ids = {}
    for entry in read_jsonl(verified(spec["ids"], roots, inputs)):
        ids[entry["id"]] = entry
    for row in rows:
        entry = ids.get(row["id"])
        if entry is None or entry["source"] != row["source"]:
            raise ValueError(
                f"{row['id']}: not in the XL r2 ids file with the same source"
            )
        check_teacher(row, teacher[row["id"]])
    if len(teacher) != len(rows):
        raise ValueError("x60 teacher file does not cover exactly the x60 rows")
    return rows, teacher, ids


def aj_production(spec, roots, inputs, wanted):
    """AutoJev targets of the production waves for the wanted ids, first file wins; a
    record for an id in two files must be identical up to float noise (reported)."""
    found: dict[str, dict[str, Any]] = {}
    source: dict[str, str] = {}
    disagree = Counter()
    for entry in spec["aj_targets"]:
        for record in read_jsonl(verified(entry, roots, inputs)):
            rid = record["id"]
            if rid not in wanted:
                continue
            if rid in found:
                a, b = found[rid]["teacher_probs"], record["teacher_probs"]
                if set(a) != set(b) or max(abs(a[k] - b[k]) for k in a) > 1e-4:
                    disagree[f"{source[rid]}|{entry['name']}"] += 1
                continue
            found[rid] = record
            source[rid] = entry["name"]
    return found, source, dict(disagree)


def prompt_cap(spec, roots, inputs) -> int:
    longest = 0
    for entry in spec["aj_prompts"]:
        with verified(entry, roots, inputs).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    longest = max(longest, len(canonical(json.loads(line))))
    return longest


def split(spec, roots, out: Path) -> dict[str, Any]:
    from v2.data.build_a0_variants import native_prompt

    inputs: dict[str, str] = {}
    rows, _, ids = load_x60(spec, roots, inputs)
    pool_of = {r["id"]: ids[r["id"]]["pool"] for r in rows}
    s_rows = [r for r in rows if human_rated(r, pool_of[r["id"]], spec)]
    covered, cover_file, disagree = aj_production(
        spec, roots, inputs, {r["id"] for r in s_rows}
    )
    for row in s_rows:
        if row["id"] in covered:
            check_teacher(row, covered[row["id"]])
    cap = prompt_cap(spec, roots, inputs)
    wave, too_long, prompts = [], [], {}
    for row in s_rows:
        if row["id"] in covered:
            continue
        prompt = native_prompt(row)
        if len(canonical(prompt)) > cap:
            too_long.append(row)
        else:
            wave.append(row)
            prompts[row["id"]] = prompt
    long_ids = {r["id"] for r in too_long}
    s_final = sorted(r["id"] for r in s_rows if r["id"] not in long_ids)
    out.mkdir(parents=True)
    wave.sort(key=lambda r: r["id"])
    manifest = {
        "schema": SCHEMA,
        "step": "split",
        "name": spec["name"],
        "inputs_sha256": inputs,
        "soft_target_rule": spec["soft_target_rule"],
        "x60_rows": len(rows),
        "prompt_cap_chars": cap,
        "s_rows": len(s_final),
        "s_rows_covered_by_production": len(covered),
        "s_rows_by_production_file": dict(Counter(cover_file.values())),
        "production_disagreements": disagree,
        "wave_rows": len(wave),
        "excluded_over_cap_rows": len(too_long),
        "files": {},
    }
    for name, lines in (
        ("S.ids.txt", "".join(f"{i}\n" for i in s_final)),
        (
            "excluded-over-cap.ids.txt",
            "".join(f"{r['id']}\n" for r in sorted(too_long, key=lambda r: r["id"])),
        ),
    ):
        (out / name).write_text(lines, encoding="utf-8")
        manifest["files"][name] = file_sha256(out / name)
    manifest["files"]["wave.rows.jsonl"] = write_lines(out / "wave.rows.jsonl", wave)
    manifest["files"]["wave.prompts.jsonl"] = write_lines(
        out / "wave.prompts.jsonl", [prompts[r["id"]] for r in wave]
    )
    by_pool: dict[str, Counter] = defaultdict(Counter)
    native = {r["id"]: ids[r["id"]]["native"] for r in rows}
    s_set = set(s_final)
    for row in rows:
        pool = pool_of[row["id"]]
        cls = (
            "S"
            if row["id"] in s_set
            else ("over_cap" if row["id"] in long_ids else "own_lux")
        )
        by_pool[pool][f"{cls}_rows"] += 1
        by_pool[pool][f"{cls}_native"] += native[row["id"]]
        if cls == "S":
            by_pool[pool][
                "S_new_wave_rows" if row["id"] not in covered else "S_production_rows"
            ] += 1
    manifest["by_pool"] = {p: dict(c) for p, c in sorted(by_pool.items())}
    manifest["s_native_tokens"] = sum(native[i] for i in s_final)
    manifest["x60_native_tokens"] = sum(native.values())
    manifest["s_rows_by_type"] = dict(
        Counter(r["task_type"] for r in s_rows if r["id"] in s_set)
    )
    return manifest


def agreement(rows, records) -> dict[str, Any]:
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        probs = records[row["id"]]["teacher_probs"]
        keys = [o["key"] for o in row["options"]]
        best = max(keys, key=lambda k: (probs[k], -keys.index(k)))
        gold = keys[row["label"]] if isinstance(row["label"], int) else row["label"]
        for key in ("all", row["task_type"]):
            hits[key][0] += best == gold
            hits[key][1] += 1
    return {k: round(a / n, 4) for k, (a, n) in sorted(hits.items())}


def ka(spec, roots, wave_targets: Path, wave_report: Path, out: Path) -> dict[str, Any]:
    inputs: dict[str, str] = {}
    rows, own, ids = load_x60(spec, roots, inputs)
    split_dir = resolve(spec["split_dir"], roots)
    split_manifest = json.loads((split_dir / "manifest.json").read_text())
    for name, sha in split_manifest["files"].items():
        if file_sha256(split_dir / name) != sha:
            raise ValueError(f"split output {name} changed since the split step")
    s_ids = set((split_dir / "S.ids.txt").read_text().split())
    wave_ids = {r["id"] for r in read_jsonl(split_dir / "wave.rows.jsonl")}
    report = json.loads(wave_report.read_text())
    if report.get("content_sha256") != file_sha256(wave_targets):
        raise ValueError("wave targets do not match their conversion report")
    if report.get("prompts_sha256") != split_manifest["files"]["wave.prompts.jsonl"]:
        raise ValueError("wave report is not for this split's prompts")
    inputs["wave_targets"] = file_sha256(wave_targets)
    inputs["wave_report"] = file_sha256(wave_report)
    produced, produced_from, disagree = aj_production(
        spec, roots, inputs, s_ids - wave_ids
    )
    wave = {r["id"]: r for r in read_jsonl(wave_targets)}
    if set(wave) != wave_ids:
        raise ValueError("wave targets do not cover exactly the wave rows")
    if set(produced) | set(wave) != s_ids or set(produced) & set(wave):
        raise ValueError("S is not covered exactly once by production + wave targets")
    teacher, origin = {}, Counter()
    by_id = {r["id"]: r for r in rows}
    for row in rows:
        rid = row["id"]
        if rid in s_ids:
            record = wave.get(rid) or produced[rid]
            origin["autojev:" + ("m6-wave" if rid in wave else produced_from[rid])] += 1
        else:
            record = own[rid]
            origin["own-lux"] += 1
        check_teacher(row, record)
        teacher[rid] = {
            "id": rid,
            "input_sha256": record["input_sha256"],
            "teacher_probs": record["teacher_probs"],
        }
    out.mkdir(parents=True)
    x60 = verified(spec["x60"]["train"], roots, inputs)
    shutil.copyfile(x60, out / "train.jsonl")
    train_sha = file_sha256(out / "train.jsonl")
    if train_sha != spec["x60"]["train"]["sha256"]:
        raise ValueError("copied train.jsonl differs from x60")
    teacher_sha = write_lines(
        out / "teacher.jsonl", [teacher[i] for i in sorted(teacher)]
    )
    s_rows = [by_id[i] for i in sorted(s_ids)]
    native = {r["id"]: ids[r["id"]]["native"] for r in rows}
    return {
        "schema": SCHEMA,
        "step": "ka",
        "name": spec["name"],
        "inputs_sha256": inputs,
        "split_manifest_sha256": file_sha256(split_dir / "manifest.json"),
        "train_rows": len(rows),
        "train_sha256": train_sha,
        "teacher_sha256": teacher_sha,
        "teacher_rows": len(teacher),
        "teacher_origin": dict(origin),
        "production_disagreements": disagree,
        "s_rows": len(s_ids),
        "s_native_tokens": sum(native[i] for i in s_ids),
        "s_gold_agreement": {
            "autojev": agreement(s_rows, teacher),
            "own_lux": agreement(s_rows, own),
        },
        "wave_report": {
            k: report.get(k)
            for k in ("wave", "rows", "content_sha256", "attestation_sha256")
        },
        "provenance_caveat": report.get("provenance_caveat"),
    }


def kh(spec, roots, tokenizer: Path, workers: int, out: Path) -> dict[str, Any]:
    inputs: dict[str, str] = {}
    rows, own, ids = load_x60(spec, roots, inputs)
    native = {r["id"]: ids[r["id"]]["native"] for r in rows}
    pool_of = {r["id"]: ids[r["id"]]["pool"] for r in rows}
    hs1 = load_partition(verified(spec["hs1"]["train"], roots, inputs), "train")
    seed = spec["seed"]
    f3 = [r for r in hs1 if r["family"] == F3]
    f1 = [r for r in hs1 if r["family"] == F1]
    lengths = dict(
        zip((r["id"] for r in f1 + f3), token_lengths(f1 + f3, tokenizer, workers))
    )
    if max(lengths.values()) > spec["max_length"]:
        raise ValueError("an HS1 block row exceeds max_length")
    f1_groups: dict[str, int] = Counter()
    for r in f1:
        f1_groups[r["group_id"]] += lengths[r["id"]]
    order = sorted(
        f1_groups, key=lambda g: hashlib.sha256(f"{seed}:f1:{g}".encode()).hexdigest()
    )
    f1_target = (
        sum(f1_groups.values())
        * spec["hs1"]["f1_share_num"]
        // spec["hs1"]["f1_share_den"]
    )
    f1_keep = set(take_prefix(order, f1_groups, f1_target))
    block = f3 + [r for r in f1 if r["group_id"] in f1_keep]
    block_tokens = sum(lengths[r["id"]] for r in block)
    x60_tokens = sum(native.values())
    kept, budget_stats = recipe_budget(
        rows,
        native,
        pool_of,
        x60_tokens - block_tokens,
        f"{seed}:keep",
        spec["keep_tolerance"],
    )
    kept_inputs = {r["input_sha256"] for r in kept}
    dup = sum(r["input_sha256"] in kept_inputs for r in block)
    if dup or len({r["input_sha256"] for r in block}) != len(block):
        raise ValueError("an HS1 block row repeats an input")
    train = sorted(kept + block, key=lambda r: r["id"])
    if len({r["id"] for r in train}) != len(train):
        raise ValueError("duplicate id in the KH TRAIN")
    partitions = {"train": train}
    for entry in spec["isolation"]:
        partitions[entry["role"]] = load_partition(
            verified(entry, roots, inputs), entry.get("split", entry["role"])
        )
    check_partition_isolation(partitions)
    registry_path = resolve(spec["denied_sources"]["c1_registry"], roots)
    keys = c1_keys(json.loads(registry_path.read_text()))
    sources = Counter(r["source"] for r in train)
    denied = {
        s: h
        for s in sources
        if (h := denied_hits(s, keys, spec["denied_sources"].get("extra", [])))
    }
    if denied:
        raise ValueError(f"denied sources in the KH TRAIN: {denied}")
    out.mkdir(parents=True)
    train_sha = write_lines(out / "train.jsonl", train)
    load_partition(out / "train.jsonl", "train")
    teacher_sha = write_lines(
        out / "teacher.jsonl",
        [own[r["id"]] for r in sorted(kept, key=lambda r: r["id"])],
    )
    kept_tokens = sum(native[r["id"]] for r in kept)
    by_type = Counter()
    for r in kept:
        by_type[r["task_type"]] += native[r["id"]]
    for r in block:
        by_type[r["task_type"]] += lengths[r["id"]]
    total = kept_tokens + block_tokens
    return {
        "schema": SCHEMA,
        "step": "kh",
        "name": spec["name"],
        "inputs_sha256": inputs,
        "keep_budget": budget_stats,
        "x60_native_tokens": x60_tokens,
        "kept_rows": len(kept),
        "kept_native_tokens": kept_tokens,
        "block": {
            "rows": len(block),
            "native_tokens": block_tokens,
            "f3_rows": len(f3),
            "f3_native_tokens": sum(lengths[r["id"]] for r in f3),
            "f1_groups_total": len(f1_groups),
            "f1_groups_kept": len(f1_keep),
            "f1_rows_kept": sum(r["group_id"] in f1_keep for r in f1),
            "f1_native_tokens_kept": sum(f1_groups[g] for g in f1_keep),
            "f1_native_tokens_total": sum(f1_groups.values()),
            "rows_by_type": dict(Counter(r["task_type"] for r in block)),
            "max_tokens": max(lengths[r["id"]] for r in block),
        },
        "train_rows": len(train),
        "train_native_tokens": total,
        "train_token_share_by_type": {
            t: round(n / total, 4) for t, n in sorted(by_type.items())
        },
        "kept_rows_by_pool": dict(
            sorted(Counter(pool_of[r["id"]] for r in kept).items())
        ),
        "train_sha256": train_sha,
        "teacher_sha256": teacher_sha,
        "teacher_rows": len(kept),
        "isolation": sorted(p for p in partitions if p != "train"),
        "source_guard": {
            "c1_registry_sha256": file_sha256(registry_path),
            "c1_keys": len(keys),
            "extra": spec["denied_sources"].get("extra", []),
            "denied": 0,
            "sources": dict(sorted(sources.items())),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("split", "ka", "kh"))
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--root", action="append", required=True, help="name=<path>")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--wave-targets", type=Path)
    ap.add_argument("--wave-report", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} exists")
    roots = {}
    for item in args.root:
        name, _, path = item.partition("=")
        roots[name] = Path(path)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.step == "split":
        manifest = split(spec, roots, args.output_dir)
    elif args.step == "ka":
        if not (args.wave_targets and args.wave_report):
            ap.error("ka needs --wave-targets and --wave-report")
        manifest = ka(spec, roots, args.wave_targets, args.wave_report, args.output_dir)
    else:
        if not args.tokenizer:
            ap.error("kh needs --tokenizer")
        manifest = kh(spec, roots, args.tokenizer, args.workers, args.output_dir)
    manifest["spec_sha256"] = file_sha256(args.spec)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    keys = (
        "step",
        "s_rows",
        "wave_rows",
        "train_rows",
        "train_sha256",
        "teacher_sha256",
        "train_native_tokens",
    )
    print(json.dumps({k: manifest[k] for k in keys if k in manifest}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
