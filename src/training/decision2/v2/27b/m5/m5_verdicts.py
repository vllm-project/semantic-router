"""Milestone 5 rule verdicts for the ~27B track (host CPU, stdlib; reads m5-gates.sh outputs).

Per sealed finalist (preregistration ``records/m5-prereg-2026-09-30.md``, "Formal runs, rules and attribution"):

* **successor rule (16:05) vs A20r's scored run:** 1. v3 paired 95% lower bound > 0; 2. human transfer not
  significantly below (the paired H interval's upper end >= 0); 3. every ``gates types`` verdict OK; 4. mlx-diag
  card-eligible Choice + Noul not significantly below (``v2.06b.m8_scorebias mlx-paired`` R4, run on node A); 5. tier
  gates: v3 >= 64.92 and H not significantly below AutoJev-27B, Eikos-27B and Jebadiah-27B; 6. no overlap exposure (0
  flagged groups in the finalist's TRAIN); 7. ``gates public231`` vs A20r is not REGRESSION. Item 8 (C1 post-key) goes
  through the eval custodian and is reported as PENDING;
* **beats AutoJev-27B:** v3 > 72.133, paired lower bound > 0 vs AutoJev-27B, H not significantly below it;
* **choice among passers:** beats-AutoJev passers by the highest lower bound vs AutoJev-27B, otherwise successor-rule
  passers by the highest lower bound vs A20r.

    python3 -m v2.27b.m5.m5_verdicts --gates DIR --finalist NAME=FORMAL_RUN [...] --output OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "decision2-27b-m5-verdicts/1"
TIER_BAR = 64.92
AUTOJEV = 72.133
TRAIN_OF = {
    "M5-FF20": ["a20"],
    "M5-FF20H": ["a20h"],
    "M5-SX": ["a20", "a20h"],
    "M5-L128": ["a20"],
}


def read(path: Path, files: dict[str, str]) -> dict[str, Any]:
    data = path.read_bytes()
    files[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def paired(path: Path, files: dict[str, str]) -> dict[str, Any]:
    record = read(path, files)
    return {
        "delta": record["point"]["delta"]["score"],
        "ci95": [record["ci95"]["low"], record["ci95"]["high"]],
        "delta_T": record["point"]["delta"]["T"],
        "delta_H": record["point"]["delta"]["H"],
        "H_ci95": [
            record["axis_ci95"]["H"]["delta"]["low"],
            record["axis_ci95"]["H"]["delta"]["high"],
        ],
        "T_ci95": [
            record["axis_ci95"]["T"]["delta"]["low"],
            record["axis_ci95"]["T"]["delta"]["high"],
        ],
        "left": record["models"]["left"],
        "right": record["models"]["right"],
    }


def finalist(
    name: str,
    run: Path,
    gates: Path,
    files: dict[str, str],
    train_of: dict[str, list[str]] = TRAIN_OF,
    exposure_prefix: str = "exposure-m5-",
) -> dict[str, Any]:
    d = gates / name
    report = read(run / "REPORT.json", files)
    v3 = report["v3"]["score"]
    vs = {
        key: paired(d / f"paired-vs-{key}.json", files)
        for key in ("A20r", "autojev27", "eikos27b", "jebadiah27b", "F1")
    }
    types = read(d / "types.json", files)["types"]
    public = read(d / "public231-vs-A20r.json", files)
    mlx_path = gates / "mlx" / f"{name}-vs-A20r.json"
    mlx = read(mlx_path, files) if mlx_path.is_file() else None
    exposure = {}
    for mix in train_of.get(name, []):
        record = read(gates / "overlap" / f"{exposure_prefix}{mix}.json", files)
        exposure[mix] = {
            "groups": len(record["groups"]),
            "methods_agree": record["methods_agree"],
        }
    items = {
        "1_v3_lower_bound_vs_A20r": {
            "ci95": vs["A20r"]["ci95"],
            "pass": vs["A20r"]["ci95"][0] > 0,
        },
        "2_H_not_below_A20r": {
            "H_ci95": vs["A20r"]["H_ci95"],
            "pass": vs["A20r"]["H_ci95"][1] >= 0,
        },
        "3_no_type_collapsed": {
            "verdicts": {t: v["verdict"] for t, v in types.items()},
            "pass": all(v["verdict"] == "OK" for v in types.values()),
        },
        "4_mlx_card_eligible_not_below_A20r": (
            {"status": "PENDING (node A mlx-paired)", "pass": None}
            if mlx is None
            else {
                "card_macro_ci95": mlx["bootstrap"]["card_macro_ci95"],
                "delta": mlx["delta"]["card_macro"],
                "type_macro_delta_report_only": mlx["delta"]["type_macro"],
                "pass": bool(mlx["R4"]["pass"]),
            }
        ),
        "5_tier_gates": {
            "v3": v3,
            "H_ci95_vs_peers": {
                k: vs[k]["H_ci95"] for k in ("autojev27", "eikos27b", "jebadiah27b")
            },
            "pass": v3 >= TIER_BAR
            and all(
                vs[k]["H_ci95"][1] >= 0
                for k in ("autojev27", "eikos27b", "jebadiah27b")
            ),
        },
        "6_no_overlap_exposure": {
            "exposure": exposure,
            "pass": bool(exposure) and all(e["groups"] == 0 for e in exposure.values()),
        },
        "7_public231_not_regression": {
            "delta": public["delta"],
            "ci95": public["ci95"],
            "verdict": public["verdict"],
            "pass": public["verdict"] != "REGRESSION",
        },
        "8_c1_postkey_guard": {
            "status": "PENDING (eval custodian hand-off)",
            "pass": None,
        },
    }
    decided = [i["pass"] for k, i in items.items() if not k.startswith("8_")]
    successor = None if None in decided else all(decided)
    beats = {
        "v3": v3,
        "vs_autojev27": vs["autojev27"],
        "pass": v3 > AUTOJEV
        and vs["autojev27"]["ci95"][0] > 0
        and vs["autojev27"]["H_ci95"][1] >= 0,
    }
    return {
        "run": str(run),
        "v3": v3,
        "T": report["v3"]["T"],
        "H": report["v3"]["H"],
        "public231": report["panels"]["public231"].get("correct"),
        "paired": vs,
        "successor_items": items,
        "successor_items_1_7": successor,
        "beats_autojev27": beats,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument(
        "--finalist", action="append", required=True, help="NAME=FORMAL_RUN"
    )
    parser.add_argument(
        "--train-of",
        action="append",
        default=[],
        help="NAME=MIX[,MIX] (M6 and later; replaces the M5 mapping)",
    )
    parser.add_argument("--exposure-prefix", default="exposure-m5-")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    train_of = TRAIN_OF
    if args.train_of:
        train_of = {}
        for spec in args.train_of:
            name, _, mixes = spec.partition("=")
            if not name or not mixes:
                raise SystemExit(f"bad --train-of {spec!r}; use NAME=MIX[,MIX]")
            train_of[name] = mixes.split(",")
    files: dict[str, str] = {}
    out = {}
    for spec in args.finalist:
        name, _, run = spec.partition("=")
        out[name] = finalist(
            name, Path(run), args.gates, files, train_of, args.exposure_prefix
        )
    beating = sorted(
        (
            n
            for n, v in out.items()
            if v["beats_autojev27"]["pass"] and v["successor_items_1_7"]
        ),
        key=lambda n: -out[n]["paired"]["autojev27"]["ci95"][0],
    )
    passing = sorted(
        (n for n, v in out.items() if v["successor_items_1_7"]),
        key=lambda n: -out[n]["paired"]["A20r"]["ci95"][0],
    )
    result = {
        "schema": SCHEMA,
        "label": "post-key same-panel rule verdicts (node B); item 8 is the eval custodian's",
        "finalists": out,
        "beats_autojev_and_successor": beating,
        "successor_items_1_7": passing,
        "choice": beating[0] if beating else (passing[0] if passing else None),
        "inputs_sha256": dict(sorted(files.items())),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                n: {
                    "v3": round(v["v3"], 3),
                    "items_1_7": v["successor_items_1_7"],
                    "beats_autojev27": v["beats_autojev27"]["pass"],
                    "lb_vs_autojev27": round(v["paired"]["autojev27"]["ci95"][0], 3),
                    "lb_vs_A20r": round(v["paired"]["A20r"]["ci95"][0], 3),
                }
                for n, v in out.items()
            }
            | {"choice": result["choice"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
