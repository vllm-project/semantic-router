"""Teacher-labelled mixture variants (M1T, M2T): attach teacher distributions and apply a target rule.

    python -m d25.vega.data.teacher_mix --mix-dir /data/d25/shared/data/v1/M1 \
        --teacher pplx11=/data/d25/vega/data/teacher/pplx11/M1/probs.jsonl --out DIR [--gold-weight 0.5]

Every row keeps its gold target in ``meta.gold_target`` and gets ``meta.teachers.<name>`` (probabilities in
options() order, null when the teacher could not score the row). Default rule (documented in the manifest):
``target = g * gold + (1 - g) * mean(available teachers)``, g = 0.5; rows without a teacher keep gold.
Shard names and row order are those of the base mixture (already shuffled). The manifest reports teacher
accuracy against gold per part and per source (hard-gold rows: argmax / p_true >= 0.5 agreement; soft-gold
rows: mean total-variation distance).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as df
from d25.vega.data.util import read_jsonl, sha256_file, write_json, write_jsonl


def load_teacher(paths: list[Path]) -> dict[str, list[float] | None]:
    out: dict[str, list[float] | None] = {}
    for path in paths:
        for line in read_jsonl(path):
            out[str(line["id"])] = line["probs"] if line.get("status") == "ok" else None
    return out


def missing(mix_dir: Path, have: list[Path], out: Path) -> int:
    """Rows of a mixture (train + dev) that the given teacher files do not cover yet."""
    done = set()
    for path in have:
        done |= {str(line["id"]) for line in read_jsonl(path)}
    base = json.loads((mix_dir / "manifest.json").read_text())
    rows = (
        row
        for name in base["files"]
        for row in read_jsonl(mix_dir / name)
        if row["id"] not in done
    )
    return write_jsonl(
        out,
        (
            {
                "id": r["id"],
                "source": r["source"],
                "family": r["family"],
                "state": r["state"],
                "question": r["question"],
                "target": r["target"],
            }
            for r in rows
        ),
    )


def agree(row: dict[str, Any], probs: list[float]) -> bool | None:
    gold = row["target"]
    if row["question"]["type"] == "noul":
        if 0.0 < gold[1] < 1.0 and abs(gold[1] - 0.5) < 0.2:
            return None
        return (probs[1] >= 0.5) == (gold[1] >= 0.5)
    if max(gold) < 0.99:
        return None
    return max(range(len(probs)), key=probs.__getitem__) == max(
        range(len(gold)), key=gold.__getitem__
    )


def tv(a: list[float], b: list[float]) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def build(
    mix_dir: Path, teachers: dict[str, list[Path]], out: Path, gold_weight: float
) -> dict[str, Any]:
    started = time.time()
    base = json.loads((mix_dir / "manifest.json").read_text())
    labels = {name: load_teacher(paths) for name, paths in teachers.items()}
    acc: dict[str, dict[str, Counter]] = {
        name: defaultdict(Counter) for name in teachers
    }
    softdist: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in teachers
    }
    counts: Counter = Counter()
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in base["files"]:
        rows = []
        for row in read_jsonl(mix_dir / name):
            gold = list(row["target"])
            meta = row["meta"]
            meta["gold_target"] = gold
            meta["teachers"] = {}
            available = []
            for teacher, table in labels.items():
                probs = table.get(row["id"], "missing")
                if probs == "missing":
                    counts[f"{teacher}:missing"] += 1
                    probs = None
                elif probs is None:
                    counts[f"{teacher}:unsupported"] += 1
                if probs is not None and len(probs) != len(gold):
                    counts[f"{teacher}:length_mismatch"] += 1
                    probs = None
                if probs is not None:
                    total = sum(probs)
                    probs = [round(p / total, 6) for p in probs]
                    available.append(probs)
                    ok = agree(row, probs)
                    split = "dev" if name == "dev.jsonl.gz" else "train"
                    for key in (
                        f"{split}:all",
                        f"{split}:part:{meta.get('part')}",
                        f"{split}:source:{row['source']}",
                    ):
                        if ok is None:
                            softdist[teacher][key].append(tv(probs, gold))
                        else:
                            acc[teacher][key]["n"] += 1
                            acc[teacher][key]["agree"] += int(ok)
                meta["teachers"][teacher] = probs
            if available:
                mean = [
                    sum(p[i] for p in available) / len(available)
                    for i in range(len(gold))
                ]
                target = [
                    gold_weight * g + (1 - gold_weight) * t for g, t in zip(gold, mean)
                ]
                s = sum(target)
                row["target"] = [round(v / s, 6) for v in target]
                drift = 1.0 - sum(row["target"])
                best = max(range(len(row["target"])), key=row["target"].__getitem__)
                row["target"][best] = round(row["target"][best] + drift, 6)
                meta["teacher"] = "+".join(sorted(labels))
                meta["target_rule"] = (
                    f"{gold_weight}*gold+{1 - gold_weight}*mean({','.join(sorted(labels))})"
                )
                counts["mixed"] += 1
            else:
                counts["gold_only"] += 1
            df.validate_row(row)
            rows.append(row)
        write_jsonl(out / name, rows)
        files[name] = {
            "rows": len(rows),
            "sha256": sha256_file(out / name),
            "bytes": (out / name).stat().st_size,
        }
    report = {}
    for teacher in teachers:
        table = {}
        for key, c in sorted(acc[teacher].items()):
            table[key] = {
                "n": c["n"],
                "accuracy": round(c["agree"] / c["n"], 4) if c["n"] else None,
            }
        for key, values in sorted(softdist[teacher].items()):
            table.setdefault(key, {})["soft_n"] = len(values)
            table[key]["soft_mean_tv"] = round(sum(values) / len(values), 4)
        report[teacher] = table
    manifest = {
        **{
            k: base[k]
            for k in (
                "corpus",
                "format",
                "seed",
                "rows",
                "composition",
                "tokens",
                "decontamination",
                "holdouts",
            )
            if k in base
        },
        "mix": base["mix"] + "T",
        "base_mix": base["mix"],
        "base_files": {k: v["sha256"] for k, v in base["files"].items()},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target_rule": f"target = {gold_weight} * gold + {1 - gold_weight} * mean(teachers with a distribution); "
        "rows without a teacher distribution keep gold; gold in meta.gold_target, teacher distributions in "
        "meta.teachers.<name> (options() order)",
        "teachers": {
            name: [
                {"probs_file": str(p), "probs_sha256": sha256_file(p)} for p in paths
            ]
            for name, paths in teachers.items()
        },
        "teacher_models": {
            "pplx11": "perplexity-ai/pplx-decider-v1.1-27b@6195aa55 (Apache-2.0), native settings (prompt pplx, "
            "noncausal full attention, checkpoint temperature), ws-eval run_rows",
            "kev27": "jaredpalmer/kev-27b@af0e6d55 (Apache-2.0)",
        },
        "counts": dict(counts),
        "files": files,
        "teacher_vs_gold": report,
        "seconds": round(time.time() - started, 1),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher",
        action="append",
        default=[],
        help="name=probs1.jsonl[,probs2.jsonl]",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gold-weight", type=float, default=0.5)
    parser.add_argument(
        "--missing-from",
        default=None,
        help="write the rows not covered by these probs files to --out",
    )
    args = parser.parse_args()
    if args.missing_from is not None:
        n = missing(
            args.mix_dir, [Path(p) for p in args.missing_from.split(",") if p], args.out
        )
        print(json.dumps({"missing_rows": n, "out": str(args.out)}))
        return
    teachers = {
        t.split("=", 1)[0]: [Path(p) for p in t.split("=", 1)[1].split(",")]
        for t in args.teacher
    }
    manifest = build(args.mix_dir, teachers, args.out, args.gold_weight)
    print(
        json.dumps({"counts": manifest["counts"], "rows": manifest["rows"]}, indent=1)
    )
    for teacher, table in manifest["teacher_vs_gold"].items():
        print(
            teacher,
            {k: v for k, v in table.items() if ":part:" in k or k.endswith(":all")},
        )


if __name__ == "__main__":
    main()
