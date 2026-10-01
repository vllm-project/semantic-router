"""Dual-scorer gate and external / frontier comparison for one IX1 run (private outputs).

    python3 -m v2.eval.ix1.compare --port port.json --kit kit/index.json --size 4B \
        --external <private frontier-gap JSON> --out compare.json

Gate (IX1 prereg §6): every Index benchmark's raw and skill agree within 1.5e-4 between the port
and kit 87d4650b, and the balanced skill within 0.01. Then, per benchmark: this run's skill x 100,
the external report's value and their difference (flagged when |Δ| > 1 skill point), and the
frontier peer's value with the weighted contribution of the difference to the headline gap
(index = Σ weight_b x skill_b with the board's area √n x gold 1.2 weights from the private file).
Index values are read from and written to private files only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NAMES = {
    25: "GPQA Diamond",
    30: "GSM8K",
    31: "ChessBench",
    32: "MuSR",
    33: "SATA-Bench",
    43: "CRUXEval",
    44: "CLadder",
    45: "HLE",
    57: "MMLU-Pro",
    58: "BBH",
    11: "ContractNLI",
    12: "ANLI",
    28: "WinoGrande",
    29: "HellaSwag",
    38: "ACOS",
    39: "FinEntity",
    40: "iSarcasmEval",
    41: "VAST",
    42: "NLI4CT",
    59: "RAGTruth",
    4: "BANKING77",
    5: "CLINC150",
    36: "BRIGHT",
    37: "Amazon ESCI",
    56: "PhishNChips",
    61: "HoVer",
    1: "BFCL",
    2: "ToolRet",
    3: "API-Bank",
    9: "Home appliances",
    62: "When2Call",
    20: "BPoMP",
    21: "Humicroedit",
    22: "POP909",
    23: "cfcolor",
    48: "ForecastBench",
    50: "Habermas",
    64: "New Yorker",
}
SCORER_TOLERANCE = 1.5e-4
HEADLINE_TOLERANCE = 0.01
EXTERNAL_FLAG = 1.0


def scorer_gate(port: dict[str, Any], kit: dict[str, Any]) -> dict[str, Any]:
    worst = {"raw": 0.0, "skill": 0.0}
    over = []
    for number in NAMES:
        a, b = port["benchmarks"][str(number)], kit["benchmarks"][str(number)]
        for key in worst:
            delta = abs(float(a[key]) - float(b[key]))
            worst[key] = max(worst[key], delta)
            if delta > SCORER_TOLERANCE:
                over.append((NAMES[number], key))
    headline = abs(port["scores"]["balanced_skill"] - kit["scores"]["balanced_skill"])
    return {
        "max_benchmark_delta": worst,
        "over_tolerance": over,
        "headline_delta": headline,
        "pass": not over and headline <= HEADLINE_TOLERANCE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    port = json.loads(args.port.read_text())
    kit = json.loads(args.kit.read_text())
    private = json.loads(args.external.read_text())
    weights = private["benchmark_index_weight"]
    external = private["dev20_external"].get(args.size)
    peer = private["frontier_peers"][args.size]
    rows = []
    ours_total = external_total = peer_total = 0.0
    for number, name in NAMES.items():
        ours = 100 * float(port["benchmarks"][str(number)]["skill"])
        weight = weights[name]
        row = {
            "catalog_id": number,
            "benchmark": name,
            "weight": weight,
            "ours": round(ours, 2),
            "coverage": port["benchmarks"][str(number)]["coverage"],
            "peer": peer["per_benchmark"][name],
            "weighted_gap_to_peer": round(
                weight * (ours - peer["per_benchmark"][name]), 3
            ),
        }
        ours_total += weight * ours
        peer_total += weight * peer["per_benchmark"][name]
        if external is not None:
            row["external"] = external[name]
            row["delta_vs_external"] = round(ours - external[name], 2)
            row["flag"] = abs(ours - external[name]) > EXTERNAL_FLAG
            external_total += weight * external[name]
        rows.append(row)
    report = {
        "schema": "ix1-compare/1",
        "label": "independent provisional 0.2.1 reproduction (private)",
        "size": args.size,
        "scorers": scorer_gate(port, kit),
        "headline": {
            "port_balanced_skill": port["scores"]["balanced_skill"],
            "kit_balanced_skill": kit["scores"]["balanced_skill"],
            "weighted_sum_check": round(ours_total, 3),
            "external": round(external_total, 3) if external is not None else None,
            "peer": {
                "name": peer["name"],
                "balanced_skill": peer["balanced_skill"],
                "weighted_sum_check": round(peer_total, 3),
            },
            "gap_to_peer": round(
                port["scores"]["balanced_skill"] - peer["balanced_skill"], 3
            ),
        },
        "flagged_vs_external": [r["benchmark"] for r in rows if r.get("flag")],
        "largest_deficits_vs_peer": [
            {k: r[k] for k in ("benchmark", "ours", "peer", "weighted_gap_to_peer")}
            for r in sorted(rows, key=lambda r: r["weighted_gap_to_peer"])[:10]
        ],
        "benchmarks": rows,
        "counts": port.get("counts"),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "size": args.size,
                "scorers_pass": report["scorers"]["pass"],
                "flagged": len(report["flagged_vs_external"]),
            }
        )
    )
    if not report["scorers"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
