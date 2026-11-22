"""Rows for policy_eval.py and candidate_eval.py from an sr-bench training export.

`vllm-sr benchmark export RUN_ID --output matrix.json` writes the response
matrix of a completed live dev run: every case with every target's graded
outcome, token usage and cost. Register one target per candidate, each a model
with its reasoning setting frozen in the target's `request_params`, and that
matrix is the outcome matrix candidate_eval.py replays.

A row is one case, with its case id as both id and task. Its attempts are keyed
by target id: the graded score as `ok`, the subject calls' cost in dollars as
`cost`, and their output tokens as `output_tokens` when usage was recorded.
A target without a recorded cost is an error, so an unpriced run cannot be read
as free.

Usage: python prepare_sr_bench.py --export matrix.json --out rows.jsonl [--targets A,B]
"""

from __future__ import annotations

import argparse
import json


def build_rows(export, targets=None):
    """One row per case of a training-matrix export, keeping the named targets or all of them."""
    if export.get("kind") != "training-matrix":
        raise ValueError(
            "expected the output of `vllm-sr benchmark export` (kind training-matrix)"
        )
    rows = []
    for case in export["cases"]:
        entries = {t["id"]: t for t in case["targets"]}
        missing = [t for t in targets or () if t not in entries]
        if missing:
            raise ValueError(f"case {case['id']!r} has no target {missing[0]!r}")
        attempts = {}
        for tid in targets or entries:
            t = entries[tid]
            ok = t["score"] if t.get("score") is not None else t.get("correct")
            if ok is None:
                raise ValueError(
                    f"case {case['id']!r}, target {tid!r}: no graded outcome"
                )
            if t.get("cost_usd") is None:
                raise ValueError(
                    f"case {case['id']!r}, target {tid!r}: no recorded cost; export a priced run"
                )
            attempt = {"ok": float(ok), "cost": float(t["cost_usd"])}
            if (t.get("usage") or {}).get("output_tokens") is not None:
                attempt["output_tokens"] = t["usage"]["output_tokens"]
            attempts[tid] = attempt
        rows.append({"id": case["id"], "task": case["id"], "attempts": attempts})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--export", required=True, help="training-matrix JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--targets", default="", help="comma-separated target ids (default: all)"
    )
    a = ap.parse_args(argv)
    with open(a.export) as f:
        export = json.load(f)
    rows = build_rows(export, [t for t in a.targets.split(",") if t] or None)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
