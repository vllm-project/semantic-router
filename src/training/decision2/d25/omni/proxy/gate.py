"""Vision-board gate for a candidate checkpoint: V_hat - margin against the live #1 and #3 Full.

    python -m d25.omni.proxy.gate --calibration calibration.json --candidate candidate.json \
        [--board vision.json] [--out gate.json]

``candidate.json`` is ``{"name": str, "public": {benchmark: skill}, "proxy": {private set: skill}}``
(unclipped local skills x 100, the same layout as one entry of the calibration measurements).
Without ``--board`` the live Space file is fetched; the board is re-audited against the aggregation
rule first. SPEC rule (the lead decides): private push when ``V_hat - margin > Full(#3)``; public flip
and submission when ``V_hat - margin > Full(#1)``. The stricter variant also requires ``P_hat`` above
that entrant's public part (omni/STATE.md decision of 10-09); both are reported.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from d25.omni.proxy import board as vb
from d25.omni.proxy.calibrate import Calibration


def decide(
    calibration: Mapping[str, Any],
    candidate: Mapping[str, Any],
    live: Mapping[str, Any],
) -> dict[str, Any]:
    cal = Calibration.from_json(calibration)
    estimate = cal.estimate(candidate["public"], candidate.get("proxy") or {})
    margin = float(calibration["summary"]["margin"])
    lower = estimate["V_hat"] - margin
    table = vb.standings(live)
    problems = vb.audit(live)
    first, third = table[0], table[2]
    rank = 1 + sum(1 for row in table if row["full"] >= lower)
    return {
        "candidate": candidate.get("name"),
        "checked_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "board_generated_utc": live.get("generated_utc"),
        "board_audit": problems,
        "P_hat": estimate["P_hat"],
        "Q_hat": estimate["Q_hat"],
        "V_hat": estimate["V_hat"],
        "margin": margin,
        "V_lower": lower,
        "rank_at_lower_bound": rank,
        "live_first": first,
        "live_third": third,
        "private_push": lower > third["full"],
        "private_push_strict": lower > third["full"]
        and estimate["P_hat"] > third["pub"],
        "public_submit": lower > first["full"],
        "public_submit_strict": lower > first["full"]
        and estimate["P_hat"] > first["pub"],
        "need_for_private_push": third["full"] + margin,
        "need_for_public_submit": first["full"] + margin,
        "private_maps": {b: m.kind for b, m in cal.private_maps.items()},
        "public_corrected": estimate["public"],
        "private_estimates": estimate["private"],
        "fallbacks": estimate["fallbacks"],
        "extrapolation": estimate["extrapolation"],
        "calibration_refs": calibration.get("refs"),
        "calibration_board": calibration.get("board"),
    }


def report(result: Mapping[str, Any]) -> str:
    first, third = result["live_first"], result["live_third"]
    lines = [
        f"candidate {result['candidate']}: V_hat {result['V_hat']:.2f} = 0.5 x P_hat {result['P_hat']:.2f}"
        f" + 0.5 x Q_hat {result['Q_hat']:.2f}; margin {result['margin']:.2f}; lower bound "
        f"{result['V_lower']:.2f} (rank {result['rank_at_lower_bound']} at the bound)",
        f"live board {result['board_generated_utc']}: #1 {first['name']} {first['full']:.2f} "
        f"(pub {first['pub']:.2f}); #3 {third['name']} {third['full']:.2f} (pub {third['pub']:.2f})",
        f"private push: {'YES' if result['private_push'] else 'no'} (strict: "
        f"{'YES' if result['private_push_strict'] else 'no'}; needs V_hat > {result['need_for_private_push']:.2f})",
        f"public + submit: {'YES' if result['public_submit'] else 'no'} (strict: "
        f"{'YES' if result['public_submit_strict'] else 'no'}; needs V_hat > {result['need_for_public_submit']:.2f})",
    ]
    if result["board_audit"]:
        lines.append(
            f"WARNING board audit: {len(result['board_audit'])} rows off the aggregation rule"
        )
    if result["fallbacks"]:
        lines.append(
            f"public-baseline fallback (proxy missing): {', '.join(result['fallbacks'])}"
        )
    for note in result["extrapolation"]:
        lines.append(f"extrapolation: {note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--board", default=None, help="vision.json path or URL (default: live Space)"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    calibration = json.loads(Path(args.calibration).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    live = vb.load(args.board)
    result = decide(calibration, candidate, live)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
