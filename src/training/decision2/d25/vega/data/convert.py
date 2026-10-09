"""Convert raw sources into training rows (decision_format row contract), one question per row.

Subcommands write DATA_ROOT/rows/<name>/part-*.jsonl.gz plus a report.json with drop counts:
    tasksource   tasksource/tasksource-jev-typed-decisions filtered-full (commercial-licence gated upstream)
    procedural   tasksource/procedural-typed-decisions (Apache-2.0), train and validation splits
    d20          vllm-sr/decision-2.0-training-data: XL r2 recipe + IB1-IB4 + HS1 + PN1 (HR2 excluded)

Ordinal ``score`` questions become ``choice`` questions over their levels (meta.orig_kind = "score"),
``noul`` targets become [p_false, p_true]. Option keys that are opaque codes (o1, c_ab12, result_3)
are replaced by their descriptions when those are unique, so the prompt shows content, not codes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as df
from d25.vega.data.fetch import D20, D20_EXTRA, D20_POOLS, PROCEDURAL, TASKSOURCE
from d25.vega.data.util import (
    DATA_ROOT,
    choice_question,
    make_row,
    noul_question,
    read_jsonl,
    write_json,
    write_jsonl,
)

OPAQUE_KEY = re.compile(
    r"^(o|c|k|n|opt|option|result|choice|cand|candidate|item|c_)[_-]?[0-9a-f]{1,10}$",
    re.I,
)
TS_DATASET = f"{TASKSOURCE[0]}@{TASKSOURCE[1][:8]}:filtered-full"
PR_DATASET = f"{PROCEDURAL[0]}@{PROCEDURAL[1][:8]}"
D20_DATASET = f"{D20[0]}@{D20[1][:8]}"


def ts_family(source: str) -> str:
    return (
        source.split("/")[0]
        if not source.startswith("multilingual/")
        else "/".join(source.split("/")[:2])
    )


# ------------------------------------------------------------------------------------------- tasksource


def convert_tasksource_shard(args: tuple[str, str]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path, out = args
    stats: Counter = Counter()
    rows = []
    for raw in pq.read_table(path).to_pylist():
        stats["in"] += 1
        kind, options, target = (
            raw["kind"],
            list(raw["options"] or []),
            list(raw["target"] or []),
        )
        meta = {
            "dataset": TS_DATASET,
            "licence": raw["license"],
            "licence_use": raw["license_use"],
            "source_split": raw["split"],
            "variant": raw["variant"],
            "group": f"ts:{raw['group_id']}",
            "orig_id": raw["id"],
            "question_id": raw["question_id"],
            "orig_kind": kind,
        }
        try:
            if kind == "noul":
                if len(target) != 1:
                    stats["drop_noul_target"] += 1
                    continue
                p = min(1.0, max(0.0, float(target[0])))
                question, tgt = noul_question(raw["question"]), [1.0 - p, p]
            elif kind in ("choice", "score"):
                if not options or len(options) != len(target):
                    stats["drop_target_mismatch"] += 1
                    continue
                if len(set(options)) != len(options):
                    stats["drop_duplicate_options"] += 1
                    continue
                if len(options) > df.MAX_OPTIONS:
                    stats["drop_over_255"] += 1
                    continue
                question, tgt = (
                    choice_question(raw["question"], dict.fromkeys(options)),
                    target,
                )
            else:
                stats[f"drop_kind_{kind}"] += 1
                continue
            meta["soft"] = any(0.0 < float(v) < 1.0 for v in tgt)
            row = make_row(
                row_id=f"ts:{raw['id']}",
                source=f"tasksource:{raw['source']}",
                family=ts_family(raw["source"]),
                state=raw["state"],
                question=question,
                target=tgt,
                meta=meta,
            )
        except (ValueError, TypeError) as error:
            stats[f"drop_invalid:{str(error)[:40]}"] += 1
            continue
        stats[f"kind_{kind}"] += 1
        rows.append(row)
    stats["out"] = write_jsonl(out, rows)
    return dict(stats)


def run_tasksource(workers: int) -> dict[str, Any]:
    shards = sorted(
        glob.glob(str(DATA_ROOT / "raw/tasksource/filtered-full/train-*.parquet"))
    )
    out_dir = DATA_ROOT / "rows/tasksource"
    jobs = [
        (path, str(out_dir / f"part-{i:05d}.jsonl.gz")) for i, path in enumerate(shards)
    ]
    with Pool(min(workers, len(jobs))) as pool:
        results = pool.map(convert_tasksource_shard, jobs)
    total: Counter = Counter()
    for result in results:
        total.update(result)
    report = {"dataset": TS_DATASET, "shards": len(shards), "stats": dict(total)}
    write_json(out_dir / "report.json", report)
    return report


# ------------------------------------------------------------------------------------------- procedural


def convert_procedural_shard(args: tuple[str, str, str]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path, split, out = args
    stats: Counter = Counter()
    rows = []
    for raw in pq.read_table(path).to_pylist():
        questions = json.loads(raw["questions"])
        answers = json.loads(raw["answers"])
        task, index = raw["task"], raw["id"].split(":")[-1]
        group = f"pr:{raw['id']}"
        for key, q in questions.items():
            stats["in"] += 1
            answer = answers.get(key)
            if not answer:
                stats["drop_no_answer"] += 1
                continue
            kind = q.get("type")
            meta = {
                "dataset": PR_DATASET,
                "licence": "apache-2.0",
                "licence_use": "commercial",
                "source_split": split,
                "group": group,
                "orig_id": raw["id"],
                "question_id": key,
                "orig_kind": kind,
                "level": raw.get("level"),
                "ts_id": f"procedural-typed-decisions-{task}:{split}:{index}:{key}",
            }
            try:
                if kind == "noul":
                    p = min(1.0, max(0.0, float(answer["noul"])))
                    question, tgt = noul_question(q.get("instructions")), [1.0 - p, p]
                elif kind == "choice":
                    criteria = q["criteria"]
                    keys = list(criteria)
                    probs = answer.get("probabilities") or {answer["choice"]: 1.0}
                    tgt = [float(probs.get(k, 0.0)) for k in keys]
                    # keys equal to their description render as the key alone
                    crit = {
                        k: (None if (v is None or v == k) else v)
                        for k, v in criteria.items()
                    }
                    question = choice_question(q.get("instructions"), crit)
                elif kind == "score":
                    levels = list(q["criteria"])
                    probs = answer.get("probabilities") or {}
                    tgt = [float(probs.get(str(i), 0.0)) for i in range(len(levels))]
                    if len(set(levels)) != len(levels):
                        stats["drop_duplicate_levels"] += 1
                        continue
                    question = choice_question(
                        q.get("instructions"), dict.fromkeys(levels)
                    )
                else:
                    stats[f"drop_kind_{kind}"] += 1
                    continue
                if sum(tgt) <= 0:
                    stats["drop_empty_target"] += 1
                    continue
                meta["soft"] = any(0.0 < v < 1.0 for v in tgt)
                row = make_row(
                    row_id=f"pr:{task}:{split}:{index}:{key}",
                    source=f"procedural:{task}",
                    family=f"procedural/{task}",
                    state=raw["state"],
                    question=question,
                    target=tgt,
                    meta=meta,
                )
            except (ValueError, TypeError, KeyError) as error:
                stats[f"drop_invalid:{str(error)[:40]}"] += 1
                continue
            stats[f"kind_{kind}"] += 1
            rows.append(row)
    stats["out"] = write_jsonl(out, rows)
    return dict(stats)


def run_procedural(workers: int) -> dict[str, Any]:
    out_dir = DATA_ROOT / "rows/procedural"
    jobs = []
    for split in ("train", "validation"):
        for i, path in enumerate(
            sorted(glob.glob(str(DATA_ROOT / f"raw/procedural/all/{split}-*.parquet")))
        ):
            jobs.append((path, split, str(out_dir / f"{split}-{i:05d}.jsonl.gz")))
    with Pool(min(workers, len(jobs))) as pool:
        results = pool.map(convert_procedural_shard, jobs)
    total: Counter = Counter()
    for result in results:
        total.update(result)
    report = {
        "dataset": PR_DATASET,
        "files": [j[2] for j in jobs],
        "stats": dict(total),
    }
    write_json(out_dir / "report.json", report)
    return report


# ------------------------------------------------------------------------------------------- decision 2.0


def d20_question(raw: dict[str, Any]) -> tuple[dict[str, Any], list[float]] | None:
    kind = raw.get("task_type")
    options = raw.get("options") or []
    label = raw.get("label")
    if not isinstance(label, int) or not 0 <= label < len(options):
        return None
    instructions = raw.get("instructions")
    if kind == "noul":
        keys = [str(o.get("key")).lower() for o in options]
        positive = {"true", "yes", "1"}
        if len(options) != 2 or not any(k in positive for k in keys):
            return None
        p_true = 1.0 if keys[label] in positive else 0.0
        return noul_question(instructions), [1.0 - p_true, p_true]
    if kind not in ("choice", "score") or not options or len(options) > df.MAX_OPTIONS:
        return None
    keys = [str(o.get("key")) for o in options]
    descs = [o.get("description") for o in options]
    if len(set(keys)) != len(keys):
        return None
    texts = [df.describe(d) if d not in (None, "") else None for d in descs]
    opaque = kind == "choice" and all(OPAQUE_KEY.match(k) for k in keys)
    if opaque and all(texts) and len(set(texts)) == len(texts):
        criteria = dict.fromkeys(texts)
    else:
        criteria = {k: d for k, d in zip(keys, descs)}
    target = [0.0] * len(options)
    target[label] = 1.0
    return choice_question(instructions, criteria), target


def convert_d20_file(
    args: tuple[str, str, str, list[str] | None, str],
) -> dict[str, Any]:
    path, pool_name, out, wanted, split = args
    wanted_set = set(wanted) if wanted is not None else None
    stats: Counter = Counter()
    rows = []
    for raw in read_jsonl(path):
        if wanted_set is not None and raw["id"] not in wanted_set:
            continue
        stats["in"] += 1
        built = d20_question(raw)
        if built is None:
            stats[f"drop_{raw.get('task_type')}_unconvertible"] += 1
            continue
        question, target = built
        meta = {
            "dataset": D20_DATASET,
            "licence": "d20-permissive (see 2.0 arm manifest licence table)",
            "licence_use": "commercial",
            "source_split": raw.get("split") or split,
            "pool": pool_name,
            "orig_source": raw.get("source"),
            "language": raw.get("language"),
            "group": f"d20:{raw.get('group_id') or raw['id']}",
            "orig_id": raw["id"],
            "orig_kind": raw.get("task_type"),
            "soft": False,
        }
        try:
            row = make_row(
                row_id=f"d20:{raw['id']}",
                source=f"d20:{pool_name}",
                family=f"d20/{raw.get('family') or pool_name}",
                state=raw.get("state"),
                question=question,
                target=target,
                meta=meta,
            )
        except (ValueError, TypeError) as error:
            stats[f"drop_invalid:{str(error)[:40]}"] += 1
            continue
        stats[f"kind_{raw.get('task_type')}"] += 1
        rows.append(row)
    stats["out"] = write_jsonl(out, rows)
    return dict(stats)


def run_d20(workers: int) -> dict[str, Any]:
    raw_dir = DATA_ROOT / "raw/d20"
    out_dir = DATA_ROOT / "rows/d20"
    wanted: dict[str, list[str]] = {}
    for item in read_jsonl(raw_dir / "m3/mixtures/xl-r2/mx-xl-full-r2.ids.jsonl"):
        wanted.setdefault(item["pool"], []).append(item["id"])
    jobs = []
    for pool_name, rel in D20_POOLS.items():
        safe = pool_name.replace(":", "-")
        jobs.append(
            (
                str(raw_dir / rel),
                pool_name,
                str(out_dir / f"xlr2-{safe}.jsonl.gz"),
                wanted.get(pool_name, []),
                "train",
            )
        )
    for name, (train_rel, dev_rel) in D20_EXTRA.items():
        jobs.append(
            (
                str(raw_dir / train_rel),
                name,
                str(out_dir / f"{name}-train.jsonl.gz"),
                None,
                "train",
            )
        )
        jobs.append(
            (
                str(raw_dir / dev_rel),
                name,
                str(out_dir / f"{name}-dev.jsonl.gz"),
                None,
                "dev",
            )
        )
    missing = sorted(set(wanted) - set(D20_POOLS))
    with Pool(min(workers, len(jobs))) as pool:
        results = pool.map(convert_d20_file, jobs)
    report = {
        "dataset": D20_DATASET,
        "xl_r2_ids": sum(len(v) for v in wanted.values()),
        "unknown_pools": missing,
        "files": {Path(j[2]).name: r for j, r in zip(jobs, results)},
        "excluded": [
            "HR2 (m5/hr2): not release-safe",
            "noncommercial-pilot: non-commercial sources",
        ],
    }
    write_json(out_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["tasksource", "procedural", "d20", "all"])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()
    for what in (
        ["tasksource", "procedural", "d20"] if args.what == "all" else [args.what]
    ):
        report = {
            "tasksource": run_tasksource,
            "procedural": run_procedural,
            "d20": run_d20,
        }[what](args.workers)
        print(
            what,
            json.dumps(
                report.get("stats")
                or {k: v for k, v in report.items() if k != "files"},
                indent=1,
            )[:3000],
            flush=True,
        )
        if what == "d20":
            for name, stats in report["files"].items():
                print("  ", name, stats, flush=True)


if __name__ == "__main__":
    main()
