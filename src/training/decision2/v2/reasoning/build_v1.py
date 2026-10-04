"""Build the reasoning wave-1 data lock: program and teacher graph problems, replay, arms, dev panel, manifest.

Arms (identical final, replay and NL rows; they differ only in node views): ``tf`` (true parent conclusions),
``tfm`` (rewired conclusions), ``f0`` (the ``tf`` node views at a placebo weight). Program problems use disjoint
seed namespaces for train and dev, and dev problems whose final input equals a train problem's are dropped. Teacher
problems are split train / dev by a hash of the released row id (dev keeps only node views: their final rows were
in the released model's training data). Every new row is 13-gram scanned against the evaluation files; a flagged
final drops its whole problem, a flagged node view drops that view.

usage: python3 -m v2.reasoning.build_v1 --out DIR --graphs G.jsonl --pool P.jsonl --replay R.jsonl
         --selflabel S0.jsonl ... --eval E1 ... [--counts arith=4500,code=4000,causal=4000,logic=5000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from training.model.data import file_sha256, validate_row

from . import gen_arith, gen_causal, gen_code, gen_logic
from .decontam import scan
from .rows import dump, final_row, node_view, problem_rows, teacher_problem

GENERATORS = {
    "arith": gen_arith,
    "code": gen_code,
    "causal": gen_causal,
    "logic": gen_logic,
}
BUILD_VERSION = "reasoning-build-v1"


def programs(name: str, count: int, namespace: str) -> list[dict[str, Any]]:
    rng = random.Random(f"{BUILD_VERSION}:{namespace}:{name}")
    out, attempts = [], 0
    while len(out) < count and attempts < count * 4:
        attempts += 1
        try:
            problem = GENERATORS[name].generate(
                rng, f"{namespace}-{name}-{attempts:06d}"
            )
        except ValueError:
            problem = None
        if problem is not None:
            out.append(problem)
    return out


def _bucket(row_id: str) -> float:
    return (
        int(
            hashlib.sha256(f"{BUILD_VERSION}:split:{row_id}".encode()).hexdigest()[:8],
            16,
        )
        / 2**32
    )


def _chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--selflabel", type=Path, nargs="+", required=True)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--counts", default="arith=4500,code=4000,causal=4000,logic=5000"
    )
    parser.add_argument("--dev-count", type=int, default=300)
    parser.add_argument("--cap", type=int, default=6)
    parser.add_argument("--long-cap", type=int, default=3)
    parser.add_argument("--long-chars", type=int, default=6000)
    parser.add_argument("--aux-total", type=float, default=0.5)
    parser.add_argument("--placebo-weight", type=float, default=1e-6)
    parser.add_argument("--dev-share", type=float, default=0.04)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {
        k: int(v) for k, v in (item.split("=") for item in args.counts.split(","))
    }

    train_problems = [
        p for name, n in counts.items() for p in programs(name, n, "train")
    ]
    train_finals = {final_row(p)["input_sha256"] for p in train_problems}
    dev_problems = [
        p
        for name in counts
        for p in programs(name, args.dev_count, "dev")
        if final_row(p)["input_sha256"] not in train_finals
    ]

    pool = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.pool.open(encoding="utf-8")
    }
    status: Counter = Counter()
    teacher_train, teacher_dev = [], []
    for line in args.graphs.open(encoding="utf-8"):
        record = json.loads(line)
        status[record.get("status")] += 1
        row = pool.get(record["id"])
        if (
            row is None
            or record.get("status") != "graph"
            or record.get("input_sha256") != row["input_sha256"]
        ):
            continue
        problem = teacher_problem(record, row)
        if problem is None:
            continue
        (teacher_dev if _bucket(row["id"]) < args.dev_share else teacher_train).append(
            problem
        )

    arms: dict[str, list[tuple[dict[str, Any], float]]] = {
        "tf": [],
        "tfm": [],
        "f0": [],
    }
    for problem in train_problems + teacher_train:
        cap = args.long_cap if _chars(problem["state"]) > args.long_chars else args.cap
        made = problem_rows(
            problem,
            cap=cap,
            aux_total=args.aux_total,
            placebo_weight=args.placebo_weight,
            seed=BUILD_VERSION,
        )
        for arm in arms:
            arms[arm] += made[arm]

    # dev panel: program finals (select split) and plain node views of program and teacher dev problems
    dev_final = [final_row(p, split="select") for p in dev_problems]
    dev_nodes = []
    for problem in dev_problems + teacher_dev:
        rng = random.Random(f"{BUILD_VERSION}:devnode:{problem['pid']}")
        for item in [
            n for n in problem["nodes"] if n["kind"] == "noul" or len(n["options"]) >= 2
        ][: args.cap]:
            view = node_view(problem, item, [], rng, problem["nodes"], "dev", "select")
            dev_nodes.append(view)

    # decontamination over every new row (program finals and all node views; released rows were audited before)
    new_rows = {
        r["id"]: r
        for arm in ("tf", "tfm")
        for r, _ in arms[arm]
        if r["source"].startswith("decision2_reasoning")
        or r["render_template"].startswith("reasoning_node")
    }
    for r in dev_final + dev_nodes:
        new_rows[r["id"]] = r
    report = scan(list(new_rows.values()), args.eval)
    flagged = report.pop("flagged")
    flagged_pids = {
        new_rows[i]["audit_metadata"]["pid"]
        for i in flagged
        if new_rows[i]["render_template"] != ""
        and new_rows[i]["audit_metadata"].get("view") == "final"
    }

    def keep(row: dict[str, Any]) -> bool:
        return (
            row["id"] not in flagged
            and row["audit_metadata"].get("pid") not in flagged_pids
        )

    replay = [json.loads(line) for line in args.replay.open(encoding="utf-8")]
    labels: dict[str, dict] = {}
    for path in args.selflabel:
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            labels[record["id"]] = record
    manifest: dict[str, Any] = {
        "build_version": BUILD_VERSION,
        "args": {k: str(v) for k, v in vars(args).items()},
        "teacher_status": dict(status),
        "decontam": report,
        "flagged_problems": len(flagged_pids),
        "files": {},
        "counts": {},
    }
    for arm, rows in arms.items():
        kept = [(r, w) for r, w in rows if keep(r)] + [(r, 1.0) for r in replay]
        for r, _ in kept:
            validate_row(r, "train")
        rows_path, weights_path = (
            args.out / f"train-{arm}.jsonl",
            args.out / f"weights-{arm}.jsonl",
        )
        manifest["counts"][arm] = dump(kept, str(rows_path), str(weights_path))
        manifest["counts"][arm]["by_template"] = dict(
            Counter(r["render_template"] for r, _ in kept).most_common()
        )
        manifest["counts"][arm]["by_type"] = dict(
            Counter(r["task_type"] for r, _ in kept)
        )
        if arm == "tf":
            ids = {r["id"]: r["input_sha256"] for r, _ in kept}
            with (args.out / "teacher-self.jsonl").open("w", encoding="utf-8") as sink:
                covered = 0
                for rid, record in labels.items():
                    if ids.get(rid) == record.get("input_sha256"):
                        sink.write(json.dumps(record) + "\n")
                        covered += 1
            manifest["counts"]["teacher_self_covered"] = covered
    for name, rows in (("rpdev-final", dev_final), ("rpdev-nodes", dev_nodes)):
        kept = [r for r in rows if keep(r)]
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as sink:
            for r in kept:
                validate_row(r, "select")
                sink.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        manifest["counts"][name] = {
            "rows": len(kept),
            "by_family": dict(Counter(r["family"] for r in kept)),
        }
    manifest["counts"]["problems"] = {
        "program_train": len(train_problems),
        "program_dev": len(dev_problems),
        "teacher_train": len(teacher_train),
        "teacher_dev": len(teacher_dev),
    }
    for path in sorted(args.out.glob("*.jsonl")):
        manifest["files"][path.name] = file_sha256(path)
    (args.out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True)
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("counts", "decontam", "teacher_status")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
