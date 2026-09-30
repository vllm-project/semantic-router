"""Milestone 8 development-only rules (DEV2.0-27B distillation top-ups of the K seeds).

* ``early``: the early stop after member 1 of a distillation arm (D_1 vs M7's matched control C_1,
  both continuations of K-s1 with seed 20260931, alpha 1, read at T = 1). The arm continues only if
  (a) its final SELECT700 family-macro accuracy is not more than 0.02 below C_1's, (b) its typed-DEV
  Noul ``rule_precedence`` count is not more than 0.01 x n below C_1's (M6's floor; the failure mode
  of M6's AutoJev teacher), (c) its PN1-dev true-paraphrase (hop) yes-rate is not more than 0.03
  below C_1's and (d) its HT-DEV v2 H is not 0.02 or more below C_1's (no FLAG against the control).
* ``recipe``: the matched-control identity check of one continuation against the control's same
  member: the trainer provenance contracts must be equal except the arm name, the teacher file hash
  and the teacher row count, and the training code, image, model source and TRAIN statistics must
  be equal. The teacher row count must equal the build's.

The alpha rule and the finalists are M7's (``lux9b.m7_rules alpha`` / ``finalists``), unchanged.
Development readouts are never release scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from lux9b import m4_rules, m6_rules

ROLE = "development-only M8 rule; not a release or post-key score"
SELECT_SLACK = 0.02
RP_SLACK = Fraction(1, 100)
HOP_SLACK = 0.03
HTDEV2_FLAG = 0.02
CONTRACT_MAY_DIFFER = ("arm", "teacher_rows")
TOP_LEVEL_EQUAL = (
    "code_sha256",
    "source_commit",
    "source_tree",
    "image_id",
    "precision",
    "model_source",
    "train_examples",
    "train_tokens",
    "train_max_tokens",
    "train_type_counts",
    "select_examples",
    "trainable_parameters",
)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def early(
    arms: dict,
    arm_key: str,
    control_key: str,
    arm_htdev2: dict,
    control_htdev2: dict,
    arm_pn1: dict,
    control_pn1: dict,
    arm_select: float,
    control_select: float,
) -> dict[str, Any]:
    if arm_pn1["reference"] != control_pn1["reference"]:
        raise ValueError("the PN1 readouts use different references")
    a, c = arms[arm_key], arms[control_key]
    rp_a, n_a = m6_rules.rp_count(a)
    rp_c, n_c = m6_rules.rp_count(c)
    if n_a != n_c:
        raise ValueError("rule_precedence n differs")
    h_a, h_c = arm_htdev2["htdev2"]["H_dev2"], control_htdev2["htdev2"]["H_dev2"]
    hop_a, hop_c = (
        arm_pn1["candidate"]["hop"]["yes"],
        control_pn1["candidate"]["hop"]["yes"],
    )
    reasons = []
    if arm_select < control_select - SELECT_SLACK:
        reasons.append(
            f"SELECT700 {arm_select:.4f} < control {control_select:.4f} - {SELECT_SLACK}"
        )
    floor = rp_c - RP_SLACK * n_c
    if rp_a < floor:
        reasons.append(
            f"rule_precedence {rp_a} < control {rp_c} - {float(RP_SLACK * n_c):g}"
        )
    if hop_a < hop_c - HOP_SLACK:
        reasons.append(f"hop yes {hop_a:.4f} < control {hop_c:.4f} - {HOP_SLACK}")
    if h_a - h_c <= -HTDEV2_FLAG:
        reasons.append(f"HT-DEV v2 {h_a:.4f} - control {h_c:.4f} <= -{HTDEV2_FLAG}")

    def side(arm, rp, h, hop, sel):
        return {
            "T": float(m4_rules.family_macro(arm)),
            "by_type": {k: v["correct"] for k, v in arm["by_type"].items()},
            "rule_precedence": rp,
            "H3_report_only": arm["H_mean"],
            "proxy_report_only": arm["proxy"],
            "H_dev2": h,
            "hop_yes": hop,
            "select": sel,
        }

    return {
        "arm": {"key": arm_key, **side(a, rp_a, h_a, hop_a, arm_select)},
        "control": {"key": control_key, **side(c, rp_c, h_c, hop_c, control_select)},
        "report_only": {
            "clean_no_yes": {
                "arm": arm_pn1["candidate"]["clean_no"]["yes"],
                "control": control_pn1["candidate"]["clean_no"]["yes"],
            },
            "pawsx6_yes": {
                "arm": arm_pn1["candidate"]["pawsx6"]["yes"],
                "control": control_pn1["candidate"]["pawsx6"]["yes"],
            },
        },
        "continue": not reasons,
        "reasons": reasons,
        "role": ROLE,
    }


def recipe(control: dict, arm: dict, teacher_rows: int) -> dict[str, Any]:
    ca, cc = arm["contract"], control["contract"]
    diffs = []
    for key in sorted(set(ca) | set(cc)):
        if key in CONTRACT_MAY_DIFFER:
            continue
        va, vc = ca.get(key), cc.get(key)
        if key == "data_sha256":
            va = {k: v for k, v in (va or {}).items() if k != "teacher"}
            vc = {k: v for k, v in (vc or {}).items() if k != "teacher"}
        if va != vc:
            diffs.append(f"contract.{key}")
    for key in TOP_LEVEL_EQUAL:
        if arm.get(key) != control.get(key):
            diffs.append(key)
    if ca.get("teacher_rows") != teacher_rows:
        diffs.append(f"teacher_rows {ca.get('teacher_rows')} != build {teacher_rows}")
    return {
        "status": "PASS" if not diffs else "FAIL",
        "differences": diffs,
        "arm": ca.get("arm"),
        "control": cc.get("arm"),
        "teacher_sha256": {
            "arm": (ca.get("data_sha256") or {}).get("teacher"),
            "control": (cc.get("data_sha256") or {}).get("teacher"),
        },
        "teacher_rows": {
            "arm": ca.get("teacher_rows"),
            "control": cc.get("teacher_rows"),
        },
        "role": ROLE,
    }


def select_accuracy(path: str | Path) -> float:
    return float(load_json(path)["family_macro_accuracy"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("early", help="member-1 early stop of a distillation arm")
    e.add_argument("--readout", required=True)
    e.add_argument("--arm-key", required=True)
    e.add_argument("--control-key", required=True)
    for side in ("arm", "control"):
        e.add_argument(f"--{side}-htdev2", required=True)
        e.add_argument(f"--{side}-pn1", required=True)
        e.add_argument(f"--{side}-select", required=True)
    r = sub.add_parser("recipe", help="matched-control identity of one continuation")
    r.add_argument("--control", required=True, help="control run provenance.json")
    r.add_argument("--arm", required=True, help="arm run provenance.json")
    r.add_argument("--teacher-rows", type=int, required=True)
    for sp in (e, r):
        sp.add_argument("--output")
    args = ap.parse_args(argv)
    if args.cmd == "early":
        out = early(
            m4_rules.load_arms([args.readout]),
            args.arm_key,
            args.control_key,
            load_json(args.arm_htdev2),
            load_json(args.control_htdev2),
            load_json(args.arm_pn1),
            load_json(args.control_pn1),
            select_accuracy(args.arm_select),
            select_accuracy(args.control_select),
        )
    else:
        out = recipe(load_json(args.control), load_json(args.arm), args.teacher_rows)
    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0 if args.cmd == "early" or out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
