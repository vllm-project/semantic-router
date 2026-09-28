"""Development-panel calibration check for CAL698 temperatures (coordinator rule 2026-09-28 23:15).

CAL698 per-type temperatures are adopted only if they do not worsen aggregate
calibration on the development panels: typed DEV (overall Brier and 10-bin
ECE, ``benchmark.score.score_suite``) and the CSS pilot (median task Brier-sum
and 15-bin max-probability ECE, ``transfer.score.score``), the scorers the
development readout and the formal report use. The decision is made on
development panels only, never on formal ones.

Stored development predictions are re-tempered offline: rows produced under
temperatures T_src are first returned to T = 1 (softmax(log q * T_src)), then
the candidate temperatures are applied (softmax(log p / T)); both equal the
native computation from the logits up to float64 rounding. Answers do not
change, which the receipt checks. Derived prediction files stay in ``--work``.

    python3 -m v2.release.dev_calibration --label L --panel-root /data/dev2/private/panels \
        --typed-dev DEV.jsonl --css-pilot PILOT.jsonl [--source-calibration SRC.json] \
        --candidate CAL698.json --work DIR --output receipt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from v2.release.examples import category
from v2.release.layout import canonical, sha_file, write_json
from v2.release.temperature_parity import question_kinds, retemper

SCHEMA = "dev2-release-dev-calibration/1"
TOLERANCE = 1e-12


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def rescale(
    rows: list[dict[str, Any]],
    kinds: dict[tuple[str, str], str],
    temperatures: dict[str, float],
    invert: bool = False,
) -> list[dict[str, Any]]:
    """Apply (or undo) per-type temperatures to every answer of every row."""
    out = []
    for row in rows:
        answers = {}
        for qid, answer in row["answers"].items():
            t = temperatures[kinds[(row["id"], qid)]]
            answers[qid] = retemper(
                answer, kinds[(row["id"], qid)], 1 / t if invert else t
            )
        clean = {k: v for k, v in row.items() if k != "calibration_sha256"}
        out.append({**clean, "answers": answers})
    return out


def answer_changes(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    other = {row["id"]: row for row in right}
    return sum(
        category(answer) != category(other[row["id"]]["answers"][qid])
        for row in left
        for qid, answer in row["answers"].items()
    )


def metrics(panel_root: Path, typed: Path, css: Path, label: str) -> dict[str, Any]:
    from benchmark.score import score_suite
    from transfer.score import score as score_css
    from v2.eval import panels

    panels.verify(panel_root, ["typed-dev", "css-pilot"])
    t = score_suite(
        panels.path(panel_root, "typed-dev", "gold"),
        typed,
        label,
        "development",
        "native",
    )
    c = score_css(panels.path(panel_root, "css-pilot", "gold"), css)["roles"]["pilot"]
    return {
        "typed_dev": {
            "T_dev": t["macro_family_accuracy"],
            "brier": t["overall"]["brier"],
            "ece_10": t["overall"]["ece_10"],
            "by_type": {
                k: {m: v.get(m) for m in ("brier", "ece_10", "correct_n", "n")}
                for k, v in t["by_type"].items()
            },
        },
        "css_pilot": {
            "H_pilot": c["median_task_macro_f1_all"],
            "median_task_brier_sum": c["median_task_brier_sum"],
            "median_task_ece_pmax_15": c["median_task_ece_pmax_15"],
        },
    }


def decide(raw: dict[str, Any], cal: dict[str, Any]) -> dict[str, Any]:
    pairs = {
        "typed_dev_brier": (raw["typed_dev"]["brier"], cal["typed_dev"]["brier"]),
        "typed_dev_ece_10": (raw["typed_dev"]["ece_10"], cal["typed_dev"]["ece_10"]),
        "css_pilot_brier": (
            raw["css_pilot"]["median_task_brier_sum"],
            cal["css_pilot"]["median_task_brier_sum"],
        ),
        "css_pilot_ece_15": (
            raw["css_pilot"]["median_task_ece_pmax_15"],
            cal["css_pilot"]["median_task_ece_pmax_15"],
        ),
    }
    worse = sorted(k for k, (r, c) in pairs.items() if c > r + TOLERANCE)
    return {
        "criteria": {k: {"raw": r, "cal698": c} for k, (r, c) in pairs.items()},
        "worsened": worse,
        "adopt": not worse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--typed-dev", type=Path, required=True)
    parser.add_argument("--css-pilot", type=Path, required=True)
    parser.add_argument("--source-calibration", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))[
        "temperature_by_type"
    ]
    source = (
        json.loads(args.source_calibration.read_text(encoding="utf-8"))[
            "temperature_by_type"
        ]
        if args.source_calibration
        else None
    )
    goldfree = args.panel_root / "goldfree"
    paths, checks = {}, {}
    for panel, stored, prompts in (
        ("typed-dev", args.typed_dev, goldfree / "typed-dev.prompts.jsonl"),
        ("css-pilot", args.css_pilot, goldfree / "css-pilot.prompts.jsonl"),
    ):
        kinds = question_kinds(_jsonl(prompts))
        rows = _jsonl(stored)
        raw = rescale(rows, kinds, source, invert=True) if source else rows
        cal = rescale(raw, kinds, candidate)
        for name, value in (("raw", raw), ("cal698", cal)):
            path = args.work / f"{panel}.{name}.predictions.jsonl"
            path.write_text(
                "".join(canonical(r) + "\n" for r in value), encoding="utf-8"
            )
            paths[(panel, name)] = path
        checks[panel] = {
            "stored_sha256": sha_file(stored),
            "slots": sum(len(r["answers"]) for r in rows),
            "answer_changes_stored_vs_raw": answer_changes(rows, raw),
            "answer_changes_raw_vs_cal698": answer_changes(raw, cal),
        }
    raw_m = metrics(
        args.panel_root,
        paths[("typed-dev", "raw")],
        paths[("css-pilot", "raw")],
        args.label,
    )
    cal_m = metrics(
        args.panel_root,
        paths[("typed-dev", "cal698")],
        paths[("css-pilot", "cal698")],
        args.label,
    )
    decision = decide(raw_m, cal_m)
    receipt = {
        "schema": SCHEMA,
        "label": args.label,
        "rule": "adopt CAL698 temperatures only if typed-DEV Brier and ECE and CSS-pilot Brier and ECE all do not worsen (coordinator 2026-09-28 23:15)",
        "candidate_sha256": sha_file(args.candidate),
        "candidate_temperatures": candidate,
        "source_calibration_sha256": (
            sha_file(args.source_calibration) if source else None
        ),
        "source_temperatures": source,
        "panels": checks,
        "raw": raw_m,
        "cal698": cal_m,
        "accuracy_unchanged": raw_m["typed_dev"]["T_dev"] == cal_m["typed_dev"]["T_dev"]
        and raw_m["css_pilot"]["H_pilot"] == cal_m["css_pilot"]["H_pilot"],
        "module_sha256": sha_file(Path(__file__)),
        **decision,
    }
    write_json(args.output, receipt)
    print(
        json.dumps(
            {
                "label": args.label,
                "adopt": receipt["adopt"],
                "worsened": receipt["worsened"],
            }
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
