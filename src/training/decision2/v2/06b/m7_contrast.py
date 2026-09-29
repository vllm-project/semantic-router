"""M7 (b) matched development contrasts: CHK deltas and the per-pair summary (stdlib only).

    python3 -m v2.06b.m7_contrast chk --aho-dir AHO --pair LABEL=TREAT_PROBS.json=CTRL_PROBS.json ...
        --output CHK-DELTA.json
    python3 -m v2.06b.m7_contrast summary --pair LABEL=TREAT=CTRL ... --contrast CONTRAST.json
        --chk CHK-DELTA.json [--root ARMS] --output SUMMARY.json

M7 prereg section 2.3 (matched contrasts, development evidence, not gated): m7-x-sk - m6-x-sk
per seed and per soup. `chk` scores both checkpoints' uncorrected CHK answers
(`m7_scorebias`: unique argmax, always-majority of the same items) per level count and
gives the paired treatment-minus-control accuracy delta with its item bootstrap (5,000 draws,
seed 20260929) plus level usage. `summary` joins `v2.06b.contrast` (DeltaP, DeltaQ = P_mean3,
10,000 draws), the READOUT Q/P and typed-DEV Score of both sides (`m6_select.candidate`) and
the CHK_5 deltas. Only aggregates are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import m6_select
from . import m7_scorebias as sb

LEVELS = (3, 4, 5)


def split_pair(item: str) -> tuple[str, str, str]:
    parts = item.split("=")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--pair {item!r}: expected LABEL=TREATMENT=CONTROL")
    return parts[0], parts[1], parts[2]


def chk_pair(aho_dir: Path, treatment: Path, control: Path) -> dict[str, Any]:
    t_items = sb.aho_items(aho_dir, "chk.jsonl", treatment)
    c_items = sb.aho_items(aho_dir, "chk.jsonl", control)
    if [i["id"] for i in t_items] != [i["id"] for i in c_items]:
        raise ValueError("treatment and control CHK items are not aligned")
    out: dict[str, Any] = {
        "treatment_state_sha256": sb.load_json(treatment)["state_sha256"],
        "control_state_sha256": sb.load_json(control)["state_sha256"],
    }
    for levels in LEVELS:
        t = [i for i in t_items if i["levels"] == levels]
        c = [i for i in c_items if i["levels"] == levels]
        if not t:
            continue
        gold = [i["y"] for i in t]
        t_pred = [sb.predicted(i["p"]) for i in t]
        c_pred = [sb.predicted(i["p"]) for i in c]
        ts, cs = sb.summary(t_pred, gold), sb.summary(c_pred, gold)
        paired = sb.paired_delta(c_pred, t_pred, gold)
        usage = sorted(
            set(ts["predicted_distribution"]) | set(cs["predicted_distribution"])
        )
        out[f"L{levels}"] = {
            "n": len(gold),
            "treatment": {k: ts[k] for k in SUMMARY_KEYS},
            "control": {k: cs[k] for k in SUMMARY_KEYS},
            "delta": {
                "accuracy": paired["delta_accuracy"],
                "accuracy_ci95": paired["ci95"],
                "changed_answers": paired["changed_answers"],
                "top_share": ts["top_share"] - cs["top_share"],
                "delta_vs_majority": ts["delta_vs_majority"] - cs["delta_vs_majority"],
                "level_usage": {
                    k: ts["predicted_distribution"].get(k, 0)
                    - cs["predicted_distribution"].get(k, 0)
                    for k in usage
                },
            },
        }
    return out


SUMMARY_KEYS = (
    "accuracy",
    "correct",
    "majority_level",
    "majority_accuracy",
    "delta_vs_majority",
    "top_value",
    "top_share",
    "predicted_distribution",
    "kappa",
)


def chk(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    sb.aho_manifest(args.aho_dir)
    report: dict[str, Any] = {
        "schema": "dev2-06b-m7-chk-delta/1",
        "label": "development evidence (not gated, never a release score)",
        "prereg": f"{sb.PREREG} section 2.3",
        "rule": "uncorrected CHK answers (unique argmax); paired item bootstrap of the "
        f"accuracy delta ({sb.DRAWS} draws, seed {sb.SEED}); treatment minus control",
        "chk_sha256": sb.file_sha256(args.aho_dir / "chk.jsonl"),
        "pairs": {},
    }
    for item in args.pair:
        label, treatment, control = split_pair(item)
        report["pairs"][label] = {
            "treatment_probs": treatment,
            "control_probs": control,
            **chk_pair(args.aho_dir, Path(treatment), Path(control)),
        }
        l5 = report["pairs"][label].get("L5", {}).get("delta")
        print(
            json.dumps(
                {
                    label: l5
                    and {k: l5[k] for k in ("accuracy", "accuracy_ci95", "top_share")}
                }
            )
        )
    sb.save(args.output, report)
    return 0


def side(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "Q": c["Q"],
        "P": c["P"],
        "T_dev": c["T_dev"],
        "H3": c["H3"],
        "typed": c["typed"],
        "score_level_counts": c["score_level_counts"],
        "score_modal_share": c["score_modal_share"],
    }


def summary(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    contrast = sb.load_json(args.contrast)["pairs"] if args.contrast else {}
    chk_pairs = sb.load_json(args.chk)["pairs"] if args.chk else {}
    rows: dict[str, Any] = {}
    for item in args.pair:
        label, treatment, control = split_pair(item)
        t = m6_select.candidate(treatment, args.root)
        c = m6_select.candidate(control, args.root)
        row: dict[str, Any] = {
            "treatment": treatment,
            "control": control,
            "readout": {"treatment": side(t), "control": side(c)},
            "delta_readout": {
                "Q": t["Q"] - c["Q"],
                "P": t["P"] - c["P"],
                "typed_score": t["typed"]["score"] - c["typed"]["score"],
                "typed_choice": t["typed"]["choice"] - c["typed"]["choice"],
                "score_modal_share": (
                    None
                    if t["score_modal_share"] is None or c["score_modal_share"] is None
                    else t["score_modal_share"] - c["score_modal_share"]
                ),
            },
        }
        paired = contrast.get(label)
        if paired is not None:
            row["paired"] = {
                key: {"delta": paired["delta"][key], "ci95": paired["ci95"][key]}
                for key in (
                    "P",
                    "P_mean3",
                    "T_dev",
                    "H_mean3",
                    "dev_score",
                    "dev_choice",
                )
                if key in paired["delta"]
            }
            row["paired"]["draws"] = paired["draws"]
            row["paired_Q_is"] = "P_mean3 = 100*sqrt(T_dev*H3)"
        if label in chk_pairs:
            row["chk5"] = chk_pairs[label].get("L5")
        rows[label] = row
        dq = row.get("paired", {}).get("P_mean3")
        dp = row.get("paired", {}).get("P")
        print(
            f"{label:16s} dQ {row['delta_readout']['Q']:+6.2f}"
            + (f" [{dq['ci95'][0]:+.2f}, {dq['ci95'][1]:+.2f}]" if dq else "")
            + (
                f"  dP {dp['delta']:+6.2f} [{dp['ci95'][0]:+.2f}, {dp['ci95'][1]:+.2f}]"
                if dp
                else ""
            )
            + f"  dScore {row['delta_readout']['typed_score']:+d}"
            + (
                f"  dCHK5 acc {row['chk5']['delta']['accuracy']:+.3f}"
                if row.get("chk5")
                else ""
            )
        )
    sb.save(
        args.output,
        {
            "schema": "dev2-06b-m7-contrast-summary/1",
            "label": "development evidence (not gated, never a release score)",
            "prereg": f"{sb.PREREG} section 2.3",
            "contrast": str(args.contrast) if args.contrast else None,
            "chk": str(args.chk) if args.chk else None,
            "pairs": rows,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("chk")
    one.add_argument("--aho-dir", type=Path, required=True)
    one.add_argument("--pair", action="append", required=True)
    one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("summary")
    two.add_argument("--pair", action="append", required=True)
    two.add_argument("--contrast", type=Path)
    two.add_argument("--chk", type=Path)
    two.add_argument("--root", type=Path, default=m6_select.DEFAULT_ROOT)
    two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"chk": chk, "summary": summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
