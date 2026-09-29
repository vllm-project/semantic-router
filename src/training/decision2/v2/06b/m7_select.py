"""M7 (b) family artifacts, eligibility and finalist order (stdlib only).

    python3 -m v2.06b.m7_select --family m7-mx-soup m7-mx-s1 m7-mx-s2 m7-mx-s3 \
        --family m7-cx-soup m7-cx-s1 m7-cx-s2 m7-cx-s3 --cross m7-mxcx-soup \
        --devchecks DIR --output SELECT.json [--root ARMS]

A `--family` lists the family soup first, then its COMPLETE seeds (a family with fewer than
two COMPLETE seeds is dropped by the caller and not listed). Names resolve as in
`m6_select` (`<root>/<name>/readout`, `full/` beside it); `DIR/<name>/devcheck.json` is the
`m7_scorebias devcheck` output of that checkpoint's uncorrected CHK probabilities.

M7 prereg sections 2.3 and 2.4, development only (never a release score):
- Q, P and the M6 amended guard (typed-DEV Choice >= 255, Score >= 101, modal Score share
  <= .90, SELECT and CAL Noul >= .85) are `m6_select.candidate`'s. A seed has no CAL
  probabilities, so its CAL Noul guard fails (as in M6).
- Family artifact (`m6_select.seedmean`): the soup if Q(soup) >= the family's seed-mean Q,
  else the median-Q seed. A soup without a readout leaves the family without an artifact.
- Eligible: a soup or family artifact that passes the guard and B-D1 (its devcheck's
  uncorrected CHK_5 modal share <= .90 and paired bootstrap 95% lower bound of accuracy
  minus always-majority accuracy > 0). A missing devcheck fails B-D1.
- Order: the cross soup first if eligible; then the highest-Q other eligible candidate
  (ties: higher H3, then name) if its Q >= the best eligible Q - 8. At most two finalists;
  none eligible means none.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import m6_select

BAND = 8.0
MAX_FINALISTS = 2
PREREG = "v2/06b/records/m7-prereg-2026-09-29.md"


def readout_exists(name: str, root: Path) -> bool:
    return (m6_select.resolve(name, root)[1] / "READOUT.json").is_file()


def bd1(name: str, devchecks: Path) -> dict[str, Any]:
    path = devchecks / name / "devcheck.json"
    if not path.is_file():
        return {"devcheck": None, "pass": False, "reason": "no devcheck"}
    check = m6_select.load_json(path)
    if check.get("label") != name:
        raise ValueError(f"{path}: label {check.get('label')!r} is not {name}")
    chk5 = (check.get("chk", {}).get("L5") or {}).get("before")
    numbers = (
        None
        if chk5 is None
        else {
            k: chk5[k]
            for k in (
                "n",
                "accuracy",
                "majority_level",
                "majority_accuracy",
                "delta_vs_majority",
                "delta_vs_majority_ci95",
                "top_value",
                "top_share",
                "predicted_distribution",
            )
        }
    )
    return {
        "devcheck": str(path),
        "state_sha256": check.get("state_sha256"),
        "chk5": numbers,
        "checks": check["B-D1"]["checks"],
        "pass": check["B-D1"]["pass"] is True,
    }


def family_artifact(
    soup: str, seeds: list[str], root: Path
) -> tuple[dict[str, Any], list[str]]:
    """The family record and the names it contributes as candidates."""
    if not readout_exists(soup, root):
        return {
            "soup": soup,
            "seeds": seeds,
            "artifact": None,
            "note": "the soup has no readout (failed); no family artifact",
        }, []
    try:
        result = m6_select.seedmean(soup, seeds, root)
    except ValueError as exc:
        return {"soup": soup, "seeds": seeds, "artifact": None, "note": str(exc)}, [
            soup
        ]
    record = {
        "soup": soup,
        "seeds": [c["name"] for c in result["seeds"]],
        "seed_Q": {c["name"]: c["Q"] for c in result["seeds"]},
        "excluded_seeds": result["excluded_seeds"],
        "seed_mean_Q": result["seed_mean_Q"],
        "seed_sd_Q": result["seed_sd_Q"],
        "soup_Q": result["soup"]["Q"],
        "median_seed": result["median_seed"],
        "median_ambiguous": result["median_ambiguous"],
        "artifact": result["artifact"],
        "artifact_is_soup": result["artifact_is_soup"],
        "rule": "soup if Q(soup) >= seed-mean Q, else the median-Q seed",
    }
    return record, [soup, result["artifact"]]


def order(candidates: dict[str, dict[str, Any]], cross: str | None) -> dict[str, Any]:
    eligible = sorted(
        (c for c in candidates.values() if c["eligible"]),
        key=lambda c: (-c["Q"], -c["H3"], c["name"]),
    )
    finalists: list[dict[str, Any]] = []
    if not eligible:
        return {
            "eligible_by_Q": [],
            "best_eligible_Q": None,
            "finalists": [],
            "verdict": "no eligible candidate: no formal run for (b)",
        }
    best = eligible[0]["Q"]
    if cross in candidates and candidates[cross]["eligible"]:
        finalists.append(
            {"name": cross, "reason": "order 1: the cross soup is eligible"}
        )
    others = [c for c in eligible if c["name"] != cross]
    if others and len(finalists) < MAX_FINALISTS:
        top = others[0]
        gap = best - top["Q"]
        if top["Q"] >= best - BAND:
            finalists.append(
                {
                    "name": top["name"],
                    "reason": f"order 2: highest-Q other eligible candidate, "
                    f"{gap:.2f} Q below the best eligible (band {BAND})",
                }
            )
        else:
            return {
                "eligible_by_Q": [c["name"] for c in eligible],
                "best_eligible_Q": best,
                "finalists": finalists,
                "not_selected": {
                    "name": top["name"],
                    "reason": f"{gap:.2f} Q below the best eligible (band {BAND})",
                },
                "verdict": f"{len(finalists)} finalist(s)",
            }
    return {
        "eligible_by_Q": [c["name"] for c in eligible],
        "best_eligible_Q": best,
        "finalists": finalists,
        "verdict": f"{len(finalists)} finalist(s)",
    }


def select(
    families: list[list[str]], cross: str | None, devchecks: Path, root: Path
) -> dict[str, Any]:
    names: list[str] = []
    records = []
    for soup, *seeds in families:
        record, contributed = family_artifact(soup, seeds, root)
        records.append(record)
        names.extend(contributed)
    cross_note = None
    if cross is not None:
        if readout_exists(cross, root):
            names.append(cross)
        else:
            cross_note = f"{cross} has no readout (failed or not built)"
    candidates: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(names):
        c = m6_select.candidate(name, root)
        c["B-D1"] = bd1(name, devchecks)
        c["eligible"] = c["guards_pass"] and c["B-D1"]["pass"]
        c["roles"] = [
            role
            for role, hit in (
                ("cross soup", name == cross),
                ("family soup", any(r["soup"] == name for r in records)),
                ("family artifact", any(r.get("artifact") == name for r in records)),
            )
            if hit
        ]
        candidates[name] = c
    result = {
        "schema": "dev2-06b-m7-select/1",
        "label": "development readout (never a release score)",
        "prereg": f"{PREREG} sections 2.3 and 2.4",
        "rules": {
            "Q": "100*sqrt(T_dev*H3), H3 = CSS-pilot three-task mean",
            "guard": "typed-DEV Choice >= 255, Score >= 101, modal Score share <= .90, "
            "SELECT and CAL Noul >= .85 (m6_select, modal90)",
            "B-D1": "uncorrected CHK_5 modal share <= .90 and paired bootstrap 95% lower "
            "bound of accuracy - always-majority > 0 (m7_scorebias devcheck)",
            "family_artifact": "soup if Q(soup) >= seed-mean Q, else the median-Q seed",
            "order": f"cross soup first if eligible; then the highest-Q other eligible "
            f"candidate within {BAND} Q of the best eligible; at most {MAX_FINALISTS}",
        },
        "families": records,
        "cross": cross,
        "cross_note": cross_note,
        "candidates": [
            {
                k: c[k]
                for k in (
                    "name",
                    "roles",
                    "Q",
                    "P",
                    "T_dev",
                    "H3",
                    "H_pilot",
                    "pilot_tasks",
                    "typed",
                    "typed_n",
                    "score_level_counts",
                    "score_modal_share",
                    "select_noul",
                    "cal_noul",
                    "in_distribution_source",
                    "ingredients",
                    "guards",
                    "guards_pass",
                    "guards_failed",
                    "B-D1",
                    "eligible",
                )
            }
            for c in candidates.values()
        ],
    }
    result.update(order(candidates, cross))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=m6_select.DEFAULT_ROOT)
    parser.add_argument(
        "--family", nargs="+", action="append", default=[], metavar="NAME"
    )
    parser.add_argument("--cross")
    parser.add_argument("--devchecks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(len(f) < 3 for f in args.family):
        parser.error("--family needs a soup and at least two seeds")
    if args.output.exists():
        raise FileExistsError(args.output)
    m6_select.SCORE_GUARD = "modal90"
    result = select(args.family, args.cross, args.devchecks, args.root)
    m6_select.write(args.output, result)
    for c in result["candidates"]:
        print(
            f"{c['name']:16s} Q {c['Q']:6.2f}  guard {'pass' if c['guards_pass'] else 'FAIL ' + ','.join(c['guards_failed'])}"
            f"  B-D1 {'pass' if c['B-D1']['pass'] else 'FAIL'}  eligible {c['eligible']}"
        )
    print("finalists: " + (", ".join(f["name"] for f in result["finalists"]) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
