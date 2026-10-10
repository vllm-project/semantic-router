"""Image-row parity of a runtime revision against the released runtime (same weights, one process per shard).

    python -m d25.vega.latency.vision run --suite SUITE --package PKG --runtime NEW --reference-runtime OLD \
        [--sample 376] [--shard 0 --shards 8] --out DIR
    python -m d25.vega.latency.vision merge --out DIR

Every row (its images as data URLs, all its questions) goes through both runtimes one request at a time.
Per question: identical probabilities, argmax / noul flips with the reference's top-2 margin, |dp|, and
whether each runtime's answer equals the row's ``expected`` answer (a plain accuracy, not the board's index).
``--sample N`` takes N rows spread over the families with at least 40 two-image rows (the runtime sample).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from d25.vega.latency.probe import load_runtime, read_rows


def correct(answer: dict, expected) -> bool | None:
    if "error" in answer:
        return False
    if answer["type"] == "noul":
        if isinstance(expected, bool):
            return (answer["noul"] >= 0.5) == expected
        if isinstance(expected, str) and expected in ("true", "false"):
            return (answer["noul"] >= 0.5) == (expected == "true")
        return None
    choice = answer.get("choice")
    if isinstance(expected, (list, tuple)):
        return choice in expected
    return None if expected is None else choice == expected


def values(answer: dict) -> list[float]:
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def cmd_run(args) -> int:
    from d25.omni.runtime.common import data_url, stratified

    suite = Path(args.suite)
    rows = read_rows(suite / "rows.jsonl.gz")
    if args.sample:
        rows = stratified(rows, args.sample, min_multi=40)
    rows = rows[args.shard :: args.shards]
    new_rt = load_runtime(args.runtime, "d3_runtime_new")
    old_rt = load_runtime(args.reference_runtime, "d3_runtime_old")
    kwargs = {"device": args.device, "verify": "none", "max_length": 1 << 20}
    new = new_rt.D3.from_pretrained(args.package, **kwargs)
    old = old_rt.D3.from_pretrained(args.package, **kwargs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {
        "rows": 0,
        "questions": 0,
        "exact": 0,
        "flips": 0,
        "flips_near_tie": 0,
        "errors": 0,
        "correct_new": 0,
        "correct_old": 0,
        "scored": 0,
        "max_abs_dp": 0.0,
    }
    diffs = []
    with open(out / f"rows-{args.shard}.jsonl", "w", encoding="utf-8") as log:
        for row in rows:
            images = [data_url(suite / p) for p in row["images"]]
            answers = {}
            for name, model in (("new", new), ("old", old)):
                try:
                    answers[name] = model.system_one(
                        state=row.get("state"),
                        questions=row["questions"],
                        images=images,
                    )["answers"]
                except Exception as exc:  # noqa: BLE001 - recorded, both must agree
                    answers[name] = {"_error": f"{type(exc).__name__}: {exc}"}
            stats["rows"] += 1
            record = {"id": row["id"], "family": row.get("family"), "questions": {}}
            if "_error" in answers["new"] or "_error" in answers["old"]:
                stats["errors"] += 1
                record["error"] = {k: v.get("_error") for k, v in answers.items()}
                log.write(json.dumps(record) + "\n")
                continue
            for key, want in answers["old"].items():
                have = answers["new"][key]
                stats["questions"] += 1
                if "error" in want or "error" in have:
                    stats["errors"] += 1
                    continue
                a, b = values(have), values(want)
                same = a == b
                stats["exact"] += same
                dp = max(abs(x - y) for x, y in zip(a, b))
                diffs.append(dp)
                stats["max_abs_dp"] = max(stats["max_abs_dp"], dp)
                top = sorted(b, reverse=True)
                margin = top[0] - (top[1] if len(top) > 1 else 0.0)
                flipped = max(range(len(a)), key=a.__getitem__) != max(
                    range(len(b)), key=b.__getitem__
                )
                stats["flips"] += flipped
                stats["flips_near_tie"] += flipped and margin <= 0.02
                expected = (
                    (row.get("expected") or {}).get(key)
                    if isinstance(row.get("expected"), dict)
                    else None
                )
                c_new, c_old = correct(have, expected), correct(want, expected)
                if c_new is not None and c_old is not None:
                    stats["scored"] += 1
                    stats["correct_new"] += c_new
                    stats["correct_old"] += c_old
                record["questions"][key] = {
                    "exact": same,
                    "abs_dp": dp,
                    "flip": flipped,
                }
            log.write(json.dumps(record) + "\n")
    stats["mean_abs_dp"] = statistics.fmean(diffs) if diffs else 0.0
    stats["fast"] = new.fast_report()
    (out / f"shard-{args.shard}.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))
    return 0


def cmd_merge(args) -> int:
    out = Path(args.out)
    shards = [json.loads(p.read_text()) for p in sorted(out.glob("shard-*.json"))]
    total = {
        k: sum(s[k] for s in shards)
        for k in (
            "rows",
            "questions",
            "exact",
            "flips",
            "flips_near_tie",
            "errors",
            "correct_new",
            "correct_old",
            "scored",
        )
    }
    total["max_abs_dp"] = max(s["max_abs_dp"] for s in shards)
    total["shards"] = len(shards)
    if total["scored"]:
        total["accuracy_new"] = round(total["correct_new"] / total["scored"], 5)
        total["accuracy_old"] = round(total["correct_old"] / total["scored"], 5)
    (out / "summary.json").write_text(json.dumps(total, indent=1) + "\n")
    print(json.dumps(total))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--suite", required=True)
    run.add_argument("--package", required=True)
    run.add_argument("--runtime", required=True)
    run.add_argument("--reference-runtime", required=True)
    run.add_argument("--sample", type=int)
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--shards", type=int, default=1)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--out", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
