"""Item-by-item answer comparison of two formal same-panel runs (counts only; stdlib).

    python3 -m v2.06b.m6_answers_diff RUN_A RUN_B [--panels typed-final,css15,public231,mlx-diag] [--json OUT]

Reads `<run>/output/<panel>.predictions.jsonl`; a panel missing there is looked up in the
sibling `<run>-mlx/output/` (the formal scripts collect `mlx-diag` in a separate run
directory). Per panel it reports items compared, answer slots, answers whose category differs
(the eval runner's `repeat` rule: Choice value, Noul side, Score argmax level), items with any
difference, rows whose answers are byte-identical, the maximum absolute drift over the numeric
answer leaves (probabilities, scores), and input-hash / status / token-usage mismatches. It
never prints or writes item ids, prompt text or gold; neither run directory is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from v2.eval.same_panel import answer_category, numeric_leaves, read_jsonl, sha_file

PANELS = ("typed-final", "css15", "public231", "mlx-diag")


def predictions(run: Path, panel: str) -> Path | None:
    for root in (run, run.with_name(run.name + "-mlx")):
        path = root / "output" / f"{panel}.predictions.jsonl"
        if path.is_file():
            return path
    return None


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def compare_panel(left: Path, right: Path) -> dict[str, Any]:
    a = {row["id"]: row for row in read_jsonl(left)}
    b = {row["id"]: row for row in read_jsonl(right)}
    common = sorted(set(a) & set(b))
    slots = changed = items_changed = identical = 0
    input_mismatch = status_mismatch = usage_mismatch = 0
    drift = 0.0
    for item in common:
        x, y = a[item], b[item]
        answers_x, answers_y = x.get("answers") or {}, y.get("answers") or {}
        identical += canonical(answers_x) == canonical(answers_y)
        input_mismatch += x.get("input_sha256") != y.get("input_sha256")
        status_mismatch += x.get("adapter_status") != y.get("adapter_status")
        usage_mismatch += canonical(x.get("usage")) != canonical(y.get("usage"))
        item_changed = False
        for key in sorted(set(answers_x) | set(answers_y)):
            slots += 1
            u, v = answers_x.get(key), answers_y.get(key)
            if answer_category(u) != answer_category(v):
                changed += 1
                item_changed = True
            nu, nv = numeric_leaves(u), numeric_leaves(v)
            for leaf in set(nu) & set(nv):
                drift = max(drift, abs(nu[leaf] - nv[leaf]))
        items_changed += item_changed
    return {
        "items_left": len(a),
        "items_right": len(b),
        "items_compared": len(common),
        "missing_left": len(set(b) - set(a)),
        "missing_right": len(set(a) - set(b)),
        "answer_slots": slots,
        "answers_differ": changed,
        "items_with_answer_difference": items_changed,
        "items_answers_byte_identical": identical,
        "max_abs_numeric_drift": drift,
        "input_sha256_mismatch": input_mismatch,
        "adapter_status_mismatch": status_mismatch,
        "usage_mismatch": usage_mismatch,
        "files_identical": sha_file(left) == sha_file(right),
        "left_sha256": sha_file(left),
        "right_sha256": sha_file(right),
    }


def diff_runs(run_a: Path, run_b: Path, panels: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "dev2-06b-m6-answers-diff/1",
        "label": "post-key same-panel",
        "left": str(run_a),
        "right": str(run_b),
        "panels": {},
        "missing_panels": [],
    }
    for panel in panels:
        left, right = predictions(run_a, panel), predictions(run_b, panel)
        if left is None or right is None:
            out["missing_panels"].append(panel)
            continue
        out["panels"][panel] = compare_panel(left, right)
    values = out["panels"].values()
    out["total"] = {
        "items_compared": sum(v["items_compared"] for v in values),
        "answer_slots": sum(v["answer_slots"] for v in values),
        "answers_differ": sum(v["answers_differ"] for v in values),
        "max_abs_numeric_drift": max(
            (v["max_abs_numeric_drift"] for v in values), default=0.0
        ),
        "missing_items": sum(v["missing_left"] + v["missing_right"] for v in values),
    }
    out["answers_equal"] = (
        bool(out["panels"])
        and not out["missing_panels"]
        and out["total"]["answers_differ"] == 0
        and out["total"]["missing_items"] == 0
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--panels", type=lambda s: s.split(","), default=list(PANELS))
    parser.add_argument(
        "--json", type=Path, help="write the full result here (new file)"
    )
    args = parser.parse_args(argv)
    result = diff_runs(args.run_a, args.run_b, args.panels)
    for panel, v in result["panels"].items():
        print(
            f"{panel:12s} compared {v['items_compared']:5d} items / {v['answer_slots']:5d} slots; "
            f"answers differ {v['answers_differ']:4d} (items {v['items_with_answer_difference']}); "
            f"max drift {v['max_abs_numeric_drift']:.3g}; missing {v['missing_left']}/{v['missing_right']}; "
            f"input/status/usage mismatch {v['input_sha256_mismatch']}/{v['adapter_status_mismatch']}/"
            f"{v['usage_mismatch']}; files identical {v['files_identical']}"
        )
    if result["missing_panels"]:
        print("panels not in both runs: " + ",".join(result["missing_panels"]))
    print(
        json.dumps(
            {"answers_equal": result["answers_equal"], **result["total"]},
            sort_keys=True,
        )
    )
    if args.json:
        with args.json.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
