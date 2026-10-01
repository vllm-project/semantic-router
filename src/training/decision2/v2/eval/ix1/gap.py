"""IX1 frontier gap analysis per size (private inputs and outputs; no values in this file).

    python3 -m v2.eval.ix1.gap --frontier <private frontier-gap JSON> --report <private external report.md> \
        --compare SIZE=compare.json [...] --out gap.json

Per size: each benchmark's weighted gap to the frontier peer (index weight x skill difference), summed
by task family, with the Decision 1.0 parent's value from the external report's appendix. A benchmark
below its parent by more than one skill point is a ``loss`` (fine-tuning regression); one that is
far below the peer while level with its parent is ``coverage`` (a family neither model was trained
for); RAGTruth is ``threshold`` (Noul P(yes) level below the fixed 0.5 cut while the ranking holds,
per the external diagnostic and the calibration study). This run's values replace the external ones
where a compare file is given.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FAMILIES = {
    "hallucination and grounding (Noul)": ["RAGTruth"],
    "phishing and security judgement": ["PhishNChips"],
    "tool-call decisions": ["When2Call", "BFCL", "API-Bank", "Home appliances"],
    "retrieval and ranking": ["ToolRet", "BRIGHT", "Amazon ESCI"],
    "stance, sarcasm and pragmatics": ["VAST", "iSarcasmEval", "Habermas"],
    "entity and aspect sentiment": ["FinEntity", "ACOS"],
    "humour and creative judgement": [
        "Humicroedit",
        "New Yorker",
        "BPoMP",
        "POP909",
        "cfcolor",
    ],
    "forecasting": ["ForecastBench"],
    "math and code reasoning": ["GSM8K", "CRUXEval"],
    "knowledge and multi-step reasoning": [
        "GPQA Diamond",
        "MMLU-Pro",
        "BBH",
        "MuSR",
        "CLadder",
        "SATA-Bench",
        "ChessBench",
        "HLE",
    ],
    "entailment and fact verification": ["ANLI", "ContractNLI", "NLI4CT", "HoVer"],
    "commonsense": ["HellaSwag", "WinoGrande"],
    "intent classification": ["BANKING77", "CLINC150"],
}
PARENT = {"0.6B": "Kai", "0.8B": "Eos", "2B": "Sol", "4B": "Nox", "9B": "Lux"}
APPENDIX_COLUMNS = ["Kai", "Eos", "Sol", "Nox", "Lux"]
LOSS = 1.0
COVERAGE_GAP = 5.0


def parents_from_report(text: str) -> dict[str, dict[str, float]]:
    """Per-benchmark 1.0 values from the appendix table (last five columns)."""
    values: dict[str, dict[str, float]] = {name: {} for name in APPENDIX_COLUMNS}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if (
            len(cells) < 16
            or cells[0] in ("Benchmark", "---")
            or set(cells[0]) <= {"-"}
        ):
            continue
        tail = cells[-5:]
        try:
            numbers = [
                float(re.sub(r"[^0-9.\-]", "", c.replace("\u2212", "-"))) for c in tail
            ]
        except ValueError:
            continue
        for name, number in zip(APPENDIX_COLUMNS, numbers):
            values[name][cells[0]] = number
    return values


def analyse(
    size: str, ours: dict[str, float], frontier: dict[str, Any], parents
) -> dict[str, Any]:
    weights = frontier["benchmark_index_weight"]
    peer = frontier["frontier_peers"][size]
    parent_name = PARENT.get(size)
    parent = parents.get(parent_name, {}) if parent_name else {}
    rows = {}
    for name, weight in weights.items():
        gap = ours[name] - peer["per_benchmark"][name]
        row = {
            "ours": ours[name],
            "peer": peer["per_benchmark"][name],
            "weighted_gap": round(weight * gap, 3),
            "parent": parent.get(name),
        }
        if name == "RAGTruth" and gap < -COVERAGE_GAP:
            row["class"] = "threshold"
        elif row["parent"] is not None and ours[name] < row["parent"] - LOSS:
            row["class"] = "loss"
        elif gap < -COVERAGE_GAP:
            row["class"] = "coverage"
        else:
            row["class"] = "level_or_ahead" if gap >= -LOSS else "small_gap"
        rows[name] = row
    families = []
    for family, names in FAMILIES.items():
        total = sum(rows[n]["weighted_gap"] for n in names)
        families.append(
            {
                "family": family,
                "weighted_gap": round(total, 3),
                "benchmarks": {n: rows[n]["class"] for n in names},
            }
        )
    families.sort(key=lambda f: f["weighted_gap"])
    headline = sum(weights[n] * ours[n] for n in weights)
    return {
        "peer": peer["name"],
        "peer_balanced_skill": peer["balanced_skill"],
        "headline_weighted": round(headline, 3),
        "gap": round(headline - peer["balanced_skill"], 3),
        "parent": parent_name,
        "families": families,
        "benchmarks": rows,
        "class_counts": {
            c: sum(r["class"] == c for r in rows.values())
            for c in ("loss", "coverage", "threshold", "small_gap", "level_or_ahead")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--compare", action="append", default=[], help="SIZE=compare.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frontier = json.loads(args.frontier.read_text())
    parents = parents_from_report(args.report.read_text())
    ours_by_size = {
        size: dict(values) for size, values in frontier["dev20_external"].items()
    }
    source = {size: "external" for size in ours_by_size}
    for spec in args.compare:
        size, _, path = spec.partition("=")
        compare = json.loads(Path(path).read_text())
        ours_by_size[size] = {r["benchmark"]: r["ours"] for r in compare["benchmarks"]}
        source[size] = "ix1"
    result = {
        size: {"source": source[size], **analyse(size, ours, frontier, parents)}
        for size, ours in ours_by_size.items()
        if size in frontier["frontier_peers"]
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for size, entry in result.items():
        top = ", ".join(
            f"{f['family']} {f['weighted_gap']}" for f in entry["families"][:3]
        )
        print(
            f"{size} [{entry['source']}] gap {entry['gap']} vs {entry['peer']}: {top}"
        )


if __name__ == "__main__":
    main()
