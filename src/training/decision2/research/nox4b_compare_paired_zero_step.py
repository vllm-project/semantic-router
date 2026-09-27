"""Compare four gold-free Nox SELECT baselines before a matched GPU trial."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

CELLS = ("control_a", "treatment_a", "control_b", "treatment_b")
GATE = 1e-4


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    identifiers = [row["id"] for row in rows]
    if len(rows) != 700 or len(set(identifiers)) != len(rows):
        raise ValueError("Each zero-step baseline needs 700 unique ordered rows")
    return rows


def probabilities(row: dict) -> dict[str, float]:
    answer = row["answer"]
    if row["task_type"] == "noul":
        probability = float(answer["noul"])
        result = {"false": 1.0 - probability, "true": probability}
    else:
        result = {
            str(key): float(value) for key, value in answer["probabilities"].items()
        }
    if not result or not all(
        math.isfinite(value) and 0 <= value <= 1 for value in result.values()
    ):
        raise ValueError("Invalid offered-option probabilities")
    return result


def compare(rows: dict[str, list[dict]]) -> dict:
    if set(rows) != set(CELLS):
        raise ValueError("The four preregistered cells are required")
    results = {}
    for left, right in itertools.combinations(CELLS, 2):
        maximum = 0.0
        category_changes = 0
        for a, b in zip(rows[left], rows[right], strict=True):
            for field in ("id", "task_type", "prompt_sha256", "token_ids_sha256"):
                if a[field] != b[field]:
                    raise ValueError(f"Native input {field} differs across cells")
            pa, pb = probabilities(a), probabilities(b)
            if pa.keys() != pb.keys():
                raise ValueError("Offered-option domains differ across cells")
            maximum = max(maximum, *(abs(pa[key] - pb[key]) for key in pa))
            category_changes += a["prediction_key"] != b["prediction_key"]
        results[f"{left}_vs_{right}"] = {
            "max_absolute_probability_drift": maximum,
            "category_changes": category_changes,
            "pass": category_changes == 0 and maximum <= GATE,
        }
    return {
        "schema_version": "decision2-nox4b-paired-zero-step/1",
        "count_per_cell": 700,
        "probability_gate": GATE,
        "pairs": results,
        "status": "PASS" if all(item["pass"] for item in results.values()) else "STOP",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for cell in CELLS:
        parser.add_argument(f"--{cell.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {cell: getattr(args, cell) for cell in CELLS}
    receipt = compare({cell: load(path) for cell, path in paths.items()})
    receipt["source_sha256"] = {cell: digest(path) for cell, path in paths.items()}
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"status": receipt["status"], "pairs": receipt["pairs"]}, sort_keys=True
        )
    )
    if receipt["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
