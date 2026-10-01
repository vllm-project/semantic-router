"""Milestone 6 results tables for the ~27B track (host CPU, stdlib only; markdown on stdout).

``devgates``: one row per candidate of each ``DEVGATES-*.json`` (gates G1-G6 with their intervals, pass, finalists).
With ``--root`` and ``--pn1-rows`` it also recomputes each candidate's PN1 report from the stored slice probabilities
(``m6_slices.pn1_report``), which carries the reported-only es / fr view of amendment 3 for every arm.

``verdicts``: one row per sealed finalist of a ``VERDICTS-*.json`` (``m5_verdicts``): v3, T, H, successor items 1-7,
beats-AutoJev and the choice. Item 8 stays the eval custodian's.

    python3 -m v2.27b.m6.m6_report devgates DEVGATES.json... [--root /data/dev2/runs/27b/m6 --pn1-rows PN1.jsonl]
    python3 -m v2.27b.m6.m6_report verdicts VERDICTS.json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

m6_slices = importlib.import_module("v2.27b.m6.m6_slices")


def ci(pair: list[float] | None, digits: int = 3) -> str:
    if not pair:
        return "n/a"
    return f"[{pair[0]:+.{digits}f}, {pair[1]:+.{digits}f}]"


def mark(ok: Any) -> str:
    return {True: "pass", False: "**FAIL**", None: "pending"}[ok]


def devgates_table(paths: list[Path]) -> list[str]:
    lines = [
        "| Arm | G1 collapse | G2 HT-DEV v2 Δ vs A20r | G3 T_dev (floor) | G4 Noul (floor) "
        "| G5 clean gold-no Δ / hop Δ | G6 B_dev Δ vs A20r | All |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    finalists = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        finalists += record["finalists"]
        for name, cand in sorted(record["candidates"].items()):
            g = cand["gates"]
            g1, g2, g3, g4, g5, g6 = (
                g["G1_collapse"],
                g["G2_htdev2_vs_A20r"],
                g["G3_typed_floor_vs_L128"],
                g["G4_noul_floor_vs_L128"],
                g["G5_pn1_guard_vs_A20r"],
                g["G6_breadth_vs_A20r"],
            )
            g5ci = g5.get("delta_ci95") or {}
            lines.append(
                f"| {name} | {mark(g1['pass'])}{'' if g1['pass'] else ' ' + '; '.join(map(str, g1['flags']))} "
                f"| {g2['delta']:+.3f} {ci(g2.get('ci95'))} {g2['verdict']} ({mark(g2['pass'])}) "
                f"| {g3['T_dev']:.4f} ({g3['T_dev_floor']:.4f}); Choice {g3['choice_accuracy']:.3f}, "
                f"Score {g3['score_accuracy']:.3f} ({mark(g3['pass'])}) "
                f"| {g4['noul_accuracy']:.4f} ({g4['floor']:.4f}) ({mark(g4['pass'])}) "
                f"| {g5['delta']['clean_no']:+.4f} {ci(g5ci.get('clean_no'), 4)} / "
                f"{g5['delta']['hop']:+.4f} {ci(g5ci.get('hop'), 4)} ({mark(g5['pass'])}) "
                f"| {g6['B_dev']:.4f}, {g6['delta']:+.4f} {ci(g6.get('delta_ci95'), 4)} ({mark(g6['pass'])}) "
                f"| {mark(cand['pass'])} |"
            )
    lines.append("")
    lines.append(f"Finalists: {', '.join(finalists) if finalists else 'none'}.")
    return lines


def heldout_table(
    root: Path, pn1_rows: Path, names: list[str], ref: str = "A20r"
) -> list[str]:
    rows = m6_slices.read_jsonl(pn1_rows)
    reference = (
        ref,
        m6_slices.read_probs(root / "slices" / ref / "probs" / "pn1.probs.jsonl", rows),
    )
    lines = [
        "| Arm | clean gold-no yes (cand / A20r) | Δ [95%] | es / fr clean gold-no Δ [95%] (n) | es / fr hop Δ (n) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in names:
        probs = m6_slices.read_probs(
            root / "slices" / name / "probs" / "pn1.probs.jsonl", rows
        )
        report = m6_slices.pn1_report(rows, (name, probs), reference)
        held = report["heldout_es_fr"] or {}
        hci = held.get("delta_ci95") or {}
        c, r = report["candidate_summary"], report["reference_summary"]
        lines.append(
            f"| {name} | {c['clean_no']['yes']:.4f} / {r['clean_no']['yes']:.4f} "
            f"| {report['delta']['clean_no']:+.4f} {ci(report['delta_ci95'].get('clean_no'), 4)} "
            f"| {held.get('delta', {}).get('clean_no', float('nan')):+.4f} {ci(hci.get('clean_no'), 4)} "
            f"({held.get('candidate', {}).get('clean_no', {}).get('n', 0)}) "
            f"| {held.get('delta', {}).get('hop', float('nan')):+.4f} "
            f"({held.get('candidate', {}).get('hop', {}).get('n', 0)}) |"
        )
    return lines


def verdicts_table(path: Path) -> list[str]:
    record = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "| Finalist | v3 (T / H) | 1. vs A20r Δ [95%] | 2. H vs A20r [95%] | 3. types | 4. mlx-diag card [95%] "
        "| 5. tier | 6. exposure | 7. public 231 | 1–7 | beats AutoJev-27B Δ [95%] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, f in sorted(record["finalists"].items()):
        items, vs = f["successor_items"], f["paired"]
        i4 = items["4_mlx_card_eligible_not_below_A20r"]
        i7 = items["7_public231_not_regression"]
        lines.append(
            f"| {name} | {f['v3']:.2f} ({f['T']:.3f} / {f['H']:.3f}) "
            f"| {vs['A20r']['delta']:+.2f} {ci(vs['A20r']['ci95'], 2)} ({mark(items['1_v3_lower_bound_vs_A20r']['pass'])}) "
            f"| {ci(vs['A20r']['H_ci95'])} ({mark(items['2_H_not_below_A20r']['pass'])}) "
            f"| {mark(items['3_no_type_collapsed']['pass'])} "
            f"| {ci(i4.get('card_macro_ci95'))} ({mark(i4['pass'])}) "
            f"| {mark(items['5_tier_gates']['pass'])} "
            f"| {mark(items['6_no_overlap_exposure']['pass'])} "
            f"| {i7['delta']:+g} {i7['verdict']} ({mark(i7['pass'])}) "
            f"| {mark(f['successor_items_1_7'])} "
            f"| {vs['autojev27']['delta']:+.2f} {ci(vs['autojev27']['ci95'], 2)} ({mark(f['beats_autojev27']['pass'])}) |"
        )
    lines.append("")
    lines.append(
        f"Items 1–7: {', '.join(record['successor_items_1_7']) or 'none'}; beats AutoJev-27B and items 1–7: "
        f"{', '.join(record['beats_autojev_and_successor']) or 'none'}; choice: {record['choice'] or 'none'}."
    )
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("devgates")
    p.add_argument("files", type=Path, nargs="+")
    p.add_argument("--root", type=Path)
    p.add_argument("--pn1-rows", type=Path)
    p = sub.add_parser("verdicts")
    p.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "verdicts":
        print("\n".join(verdicts_table(args.file)))
        return
    print("\n".join(devgates_table(args.files)))
    if args.root and args.pn1_rows:
        names = sorted(
            {
                n
                for f in args.files
                for n in json.loads(f.read_text(encoding="utf-8"))["candidates"]
            }
        )
        print()
        print("\n".join(heldout_table(args.root, args.pn1_rows, names)))


if __name__ == "__main__":
    main()
