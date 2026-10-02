"""Decoder M17 development gates, anchor decision and finalists (prereg dec-m17-prereg-2026-10-02.md, "Development
readouts and gates").

Each point is gated against the released LH read on node F (4b-LH-f):
  1-6. M16's gates without its old MLX-DEV guard: m11_rules.gate with M16's tier argument (M8 eligibility type /
       family / Noul floors; Score5-typed-DEV floor, m8s amendment 3 when the reference COLLAPSEs; HT-DEV v2 not FLAG;
       retention macro CI upper >= 0; hs1-dev false-yes <= reference + .10) and m14_rules.breadth (transfer macro
       delta >= 0);
  7.   MLX-DEV2: the card-eligible (Choice + Noul) paired delta's 95% CI upper bound >= 0 against the reference's
       node-F read (v2.eval.mlx_dev2 compare -> <lines-root>/<point>/mlx2cmp/<point>.mlx2.json).
  The old MLX-DEV compare (<lines-root>/<point>/mlxcmp/<point>.mlxdev.json) is reported, never gating.
Modes (each runs once; the output must not exist):
  arms   the two arms -> per-arm verdicts, the number eligible, whether anchors are built (fewer than two arms
         eligible) and the better arm among the arms with readouts: more gates passed; then the larger transfer macro
         delta (4 decimals); then the larger MLX-DEV2 card delta.
  final  the arms' rows from the arms result (never re-evaluated) plus the anchors' gates: eligible = gates 1-7; rank
         by the transfer macro delta (4 decimals, larger first), then HT-DEV v2 GAIN, then the smaller alpha (an arm is
         alpha 1); finalist 1 = the top point; finalist 2 = the top eligible point of the other arm line, else the next
         eligible point; at most two.
Development only; never a release or post-key score; never v3, C1, mlx-diag, public 231 or Index rows.

usage: m17_rules.py arms --lines-root L --mlx2-root P --readout R.json --point X=REF [...] --output OUT
       m17_rules.py final --lines-root L --mlx2-root P --readout R.json --arms-result A.json [--point X=REF ...] \
         --output OUT
<P>/<name>/mlx-dev2.predictions.jsonl are the MLX-DEV2 predictions (node A's private copies).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "dec-m17-rules/1"
MAX_FINALISTS = 2
ARMS = ("4b-LHS10SD", "4b-LHS17SD")
ANCHOR = re.compile(r"^(?P<line>4b-LHS1[07]SD)-a(?P<pct>33|67)$")
ALPHA = {"33": 1 / 3, "67": 2 / 3}
GATES = 7
OPS = Path(__file__).resolve().parents[1]


def load(rel: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, OPS / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_alpha(point: str) -> tuple[str, float]:
    if point in ARMS:
        return point, 1.0
    match = ANCHOR.match(point)
    if not match:
        raise ValueError(f"{point}: M17 points are the arms or <arm>-a33 / -a67")
    return match["line"], ALPHA[match["pct"]]


def gate_of(reason: str) -> int:
    """The prereg gate (1-7) a reason belongs to."""
    if reason.startswith("MLX-DEV2"):
        return 7
    if reason.startswith("breadth"):
        return 6
    if reason.startswith("yes-bias"):
        return 5
    if reason.startswith("retention"):
        return 4
    if reason.startswith("HT-DEV"):
        return 3
    if reason.startswith(("Score floor", "Score5-typed-DEV")):
        return 2
    return 1


def mlx2_guard(
    lines: Path, mlx2_root: Path, point: str, ref: str
) -> tuple[dict[str, Any] | None, list[str]]:
    path = lines / point / "mlx2cmp" / f"{point}.mlx2.json"
    if not path.is_file():
        return None, [f"MLX-DEV2: no compare of {point} against {ref}"]
    cmp = json.loads(path.read_text())
    ref_pred = mlx2_root / ref / "mlx-dev2.predictions.jsonl"
    if (
        cmp.get("candidate_name") != point
        or cmp.get("reference_name") != ref
        or not ref_pred.is_file()
        or sha256(ref_pred) != cmp["predictions_sha256"]["reference"]
    ):
        return None, [f"MLX-DEV2: the compare of {point} is not against {ref}'s read"]
    card = cmp["card_eligible"]
    row = {
        "card_delta": card["delta"],
        "card_ci95": [card["ci95"]["low"], card["ci95"]["high"]],
        "per_type": {
            t: {"delta": v["delta"], "ci95": [v["ci95"]["low"], v["ci95"]["high"]]}
            for t, v in cmp["per_type"].items()
        },
        "guard_pass": cmp["guard_pass"],
    }
    if card["ci95"]["high"] < 0 or not cmp["guard_pass"]:
        return row, [
            f"MLX-DEV2 guard: card delta {card['delta']:+.4f}, upper bound {card['ci95']['high']:+.4f} < 0 vs {ref}"
        ]
    return row, []


def old_mlx(lines: Path, point: str) -> dict[str, Any] | None:
    path = lines / point / "mlxcmp" / f"{point}.mlxdev.json"
    if not path.is_file():
        return None
    cmp = json.loads(path.read_text())
    return {
        k: {"diff": v["diff"], "ci95": v["ci95"]}
        for k, v in cmp["metrics"].items()
        if k in ("noul_ml", "choice_ml", "score_ml", "m_dev")
    }


def evaluate(
    mods: dict[str, Any],
    arms: dict[str, Any],
    lines: Path,
    mlx2_root: Path,
    point: str,
    ref: str,
) -> dict[str, Any]:
    if not ref.startswith("4b-"):
        raise ValueError(f"{point}={ref}: the reference must be a 4B point")
    line, alpha = line_alpha(point)
    row = mods["m11"].gate(
        mods["m10"], mods["m8"], mods["m8s"], arms, lines / "diag", "-", point, ref
    )
    row["line"], row["alpha"] = line, alpha
    row["ib_dev"], extra = mods["m14"].breadth(lines / "diag", point, ref)
    row["mlx_dev2"], guard = mlx2_guard(lines, mlx2_root, point, ref)
    row["old_mlx_dev"] = old_mlx(lines, point)
    row["reasons"] = row["reasons"] + extra + guard
    row["failed_gates"] = sorted({gate_of(r) for r in row["reasons"]})
    row["gates_passed"] = GATES - len(row["failed_gates"])
    row["eligible"] = not row["reasons"]
    return row


def rank_key(row: dict[str, Any]) -> tuple:
    delta = (row.get("ib_dev") or {}).get("transfer_delta")
    return (
        -round(delta if delta is not None else float("-inf"), 4),
        (row.get("htdev2") or {}).get("verdict") != "GAIN",
        row["alpha"],
    )


def better_key(row: dict[str, Any]) -> tuple:
    delta = (row.get("ib_dev") or {}).get("transfer_delta")
    card = (row.get("mlx_dev2") or {}).get("card_delta")
    return (
        -row["gates_passed"],
        -round(delta if delta is not None else float("-inf"), 4),
        -(card if card is not None else float("-inf")),
    )


def pick(rows: list[dict[str, Any]]) -> list[str]:
    passing = sorted((r for r in rows if r["eligible"]), key=rank_key)
    if not passing:
        return []
    first, rest = passing[0], passing[1:]
    other = [r for r in rest if r["line"] != first["line"]]
    second = (other or rest)[:1]
    return [r["point"] for r in [first, *second]][:MAX_FINALISTS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("arms", "final"))
    parser.add_argument("--lines-root", type=Path, required=True)
    parser.add_argument("--mlx2-root", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--point", action="append", default=[], help="X=REF")
    parser.add_argument("--arms-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; the rules run once")
    mods = {
        "m14": load("m14/m14_rules.py", "m14_rules"),
        "m11": load("m11/m11_rules.py", "m11_rules"),
        "m10": load("m10/m10_rules.py", "m10_rules"),
        "m8": load("m8/m8_rules.py", "m8_rules"),
        "m8s": load("m8s/m8s_rules.py", "m8s_rules"),
    }
    readout = json.loads(args.readout.read_text())["arms"]
    specs = [s.split("=", 1) for s in args.point]
    if args.mode == "arms":
        if not specs or not all(p in ARMS for p, _ in specs):
            raise ValueError("arms mode takes the M17 arms only")
        rows = [
            evaluate(mods, readout, args.lines_root, args.mlx2_root, p, ref)
            for p, ref in specs
        ]
        eligible = [r["point"] for r in rows if r["eligible"]]
        better = sorted(rows, key=better_key)[0]["point"] if rows else None
        result = {
            "schema": SCHEMA,
            "mode": "arms",
            "points": rows,
            "eligible": eligible,
            "anchors_needed": len(eligible) < 2,
            "better_arm": better,
            "rule": "gates 1-7 (M16's gates with MLX-DEV2 card CI upper >= 0 replacing the old MLX-DEV guard); "
            "anchors iff fewer than two arms are eligible; better arm: more gates passed, then the larger transfer "
            "macro delta (4 decimals), then the larger MLX-DEV2 card delta",
        }
    else:
        if args.arms_result is None:
            parser.error("final needs --arms-result")
        prior = json.loads(args.arms_result.read_text())
        rows = list(prior["points"])
        for p, ref in specs:
            line, _ = line_alpha(p)
            if p in ARMS:
                raise ValueError(
                    f"{p}: the arms' verdicts come from {args.arms_result}"
                )
            if not prior["anchors_needed"] or line != prior["better_arm"]:
                raise ValueError(
                    f"{p}: anchors are only of the better arm, and only when needed"
                )
            rows.append(
                evaluate(mods, readout, args.lines_root, args.mlx2_root, p, ref)
            )
        result = {
            "schema": SCHEMA,
            "mode": "final",
            "arms_result_sha256": sha256(args.arms_result),
            "points": rows,
            "finalists": pick(rows),
            "rule": "eligible = gates 1-7; rank: transfer macro delta (4 decimals, larger first), then HT-DEV v2 "
            "GAIN, then the smaller alpha (arm = 1); finalist 2 = the top eligible point of the other arm line, else "
            "the next eligible point; at most two",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {r["point"]: r["reasons"] for r in result["points"]}
    extra = (
        {"anchors_needed": result["anchors_needed"], "better_arm": result["better_arm"]}
        if args.mode == "arms"
        else {"finalists": result["finalists"]}
    )
    print(json.dumps({**extra, "points": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
