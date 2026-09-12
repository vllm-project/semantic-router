#!/usr/bin/env python3
"""Validate and render the archived B200 context-length sweep."""

import csv
import math
from pathlib import Path

GPU_COUNT = 8
STUDY_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = STUDY_ROOT / "data" / "b200_table1_measurements" / "final_results.csv"
REQUIRED_COLUMNS = {
    "context_tokens",
    "context_k",
    "measured_b200_nmax",
    "decode_tok_s",
    "per_gpu_decode_power_w",
    "measured_decode_tok_per_w",
    "measured_8gpu_decode_tok_per_w",
}
ROUNDING_TOLERANCE = 0.011


def load_rows():
    """Load the archived rows and validate the CSV schema."""
    with DATA_PATH.open(newline="", encoding="utf-8") as data_file:
        reader = csv.DictReader(data_file)
        fieldnames = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing required CSV columns: {names}")
        rows = list(reader)

    if not rows:
        raise ValueError("The measurement CSV is empty")
    return rows


def validate_row(row, seen_contexts):
    """Validate one row and return values used by the rendered table."""
    context_tokens = int(row["context_tokens"])
    context_k = int(row["context_k"])
    nmax = int(row["measured_b200_nmax"])
    decode_tok_s = float(row["decode_tok_s"])
    per_gpu_power_w = float(row["per_gpu_decode_power_w"])
    reported_paper_tpw = float(row["measured_decode_tok_per_w"])
    reported_system_tpw = float(row["measured_8gpu_decode_tok_per_w"])

    if context_tokens in seen_contexts:
        raise ValueError(f"Duplicate context_tokens value: {context_tokens}")
    seen_contexts.add(context_tokens)

    if context_tokens <= 0 or context_tokens % 1024:
        raise ValueError(f"Invalid context_tokens value: {context_tokens}")
    if context_k != context_tokens // 1024:
        raise ValueError(f"context_k mismatch for {context_tokens}: got {context_k}")
    if nmax <= 0 or decode_tok_s <= 0 or per_gpu_power_w <= 0:
        raise ValueError(f"Non-positive measurement at {context_k}K")

    calculated_paper_tpw = decode_tok_s / per_gpu_power_w
    calculated_system_tpw = calculated_paper_tpw / GPU_COUNT
    if not math.isclose(
        calculated_paper_tpw,
        reported_paper_tpw,
        abs_tol=ROUNDING_TOLERANCE,
    ):
        raise ValueError(f"Paper-normalized tok/W mismatch at {context_k}K")
    if not math.isclose(
        calculated_system_tpw,
        reported_system_tpw,
        abs_tol=ROUNDING_TOLERANCE,
    ):
        raise ValueError(f"8-GPU system tok/W mismatch at {context_k}K")

    return (
        context_tokens,
        context_k,
        nmax,
        decode_tok_s,
        per_gpu_power_w,
        reported_paper_tpw,
        reported_system_tpw,
    )


def main():
    """Validate all rows and print the measured table as Markdown."""
    seen_contexts = set()
    rows = sorted(
        (validate_row(row, seen_contexts) for row in load_rows()),
        key=lambda values: values[0],
    )

    print("| Context | nmax | Decode tok/s | Mean GPU W | Paper tok/W | System tok/W |")
    print("|---:|---:|---:|---:|---:|---:|")
    for _, context_k, nmax, tok_s, power_w, paper_tpw, system_tpw in rows:
        print(
            f"| {context_k}K | {nmax} | {tok_s:.2f} | {power_w:.2f} "
            f"| {paper_tpw:.2f} | {system_tpw:.2f} |"
        )

    print(f"\nValidated {len(rows)} rows from {DATA_PATH.relative_to(STUDY_ROOT)}.")


if __name__ == "__main__":
    main()
