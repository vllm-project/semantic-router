"""Zero-step native-shape check for an official encoder with a fresh readout.

Synthetic System One requests only (no benchmark or training text): Choice
with 2, 10 and 255 options, Noul with and without criteria, Score with 3 and
10 levels, and one long state near 6K tokens. Requires finite normalized
distributions over exactly the offered candidates, no truncation, stable
outputs under padding, and records loaded parameters, latency and memory.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from . import encoder as enc
from . import kai8k
from .common import import_bundle, write_json


def requests() -> list[tuple[str, Any, dict[str, Any]]]:
    long_state = " ".join(
        f"Ledger entry {i}: account A{i % 17} moved {i % 9} units to B{i % 11}."
        for i in range(700)
    )
    return [
        (
            "choice-2",
            "The shipment arrived late but complete.",
            {
                "q": {
                    "type": "choice",
                    "instructions": "What happened?",
                    "criteria": {"late": "It was late", "lost": "It was lost"},
                }
            },
        ),
        (
            "choice-10",
            "Pick the tenth item.",
            {
                "q": {
                    "type": "choice",
                    "instructions": "Which item?",
                    "criteria": {f"k{i}": f"Item number {i}" for i in range(10)},
                }
            },
        ),
        (
            "choice-255",
            "One of many options is right.",
            {
                "q": {
                    "type": "choice",
                    "instructions": "Which option?",
                    "criteria": {f"o{i}": f"Option {i} text" for i in range(255)},
                }
            },
        ),
        (
            "noul-default",
            {"temperature": 31, "unit": "C"},
            {"q": {"type": "noul", "instructions": "Is it above 30 degrees?"}},
        ),
        (
            "noul-criteria",
            "The door is open.",
            {
                "q": {
                    "type": "noul",
                    "instructions": "Is the door closed?",
                    "criteria": {"true": "closed", "false": "open"},
                }
            },
        ),
        (
            "score-3",
            "Service was fine.",
            {
                "q": {
                    "type": "score",
                    "instructions": "Rate the service.",
                    "criteria": ["bad", "fine", "great"],
                }
            },
        ),
        (
            "score-10",
            "Nine of ten checks passed.",
            {
                "q": {
                    "type": "score",
                    "instructions": "How many checks passed?",
                    "criteria": [f"{i} passed" for i in range(10)],
                }
            },
        ),
        (
            "long-choice",
            long_state,
            {
                "q": {
                    "type": "choice",
                    "instructions": "Which account moved most?",
                    "criteria": {"a": "A1", "b": "A2", "c": "A3"},
                }
            },
        ),
        (
            "multi",
            "Two questions share this state.",
            {
                "x": {"type": "noul", "instructions": "Is there a state?"},
                "y": {
                    "type": "choice",
                    "instructions": "How many questions?",
                    "criteria": {"one": "1", "two": "2"},
                },
            },
        ),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    import_bundle(args.bundle)
    started = time.monotonic()
    model, packer, metadata = enc.from_official(args.source_path, args.source)
    model.to("cuda:0").eval()
    torch.cuda.reset_peak_memory_stats()
    cases = []
    for name, state, questions in requests():
        rows = kai8k.request_rows(state, questions)
        encoded = [packer.encode(r) for r in rows]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs = enc.probabilities(model, packer, rows, device="cuda:0")
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000
        again = enc.probabilities(model, packer, rows, device="cuda:0")
        repeat = max(abs(a - b) for x, y in zip(probs, again) for a, b in zip(x, y))
        ok = all(
            len(p) == len(e["candidate_ids"])
            and all(math.isfinite(v) and v >= 0 for v in p)
            and abs(sum(p) - 1) < 1e-4
            for p, e in zip(probs, encoded)
        )
        cases.append(
            {
                "case": name,
                "questions": len(rows),
                "candidates": [len(e["candidate_ids"]) for e in encoded],
                "input_tokens": [e["input_tokens"] for e in encoded],
                "valid_distribution": ok,
                "repeat_max_abs_drift": repeat,
                "latency_ms": latency,
            }
        )
    padded_rows = kai8k.request_rows(*requests()[0][1:]) + kai8k.request_rows(
        *requests()[7][1:]
    )
    alone = enc.probabilities(
        model, packer, padded_rows[:1], device="cuda:0", batch_size=1
    )[0]
    batched = enc.probabilities(
        model, packer, padded_rows, device="cuda:0", batch_size=8
    )[0]
    padding_drift = max(abs(a - b) for a, b in zip(alone, batched))
    result = {
        "status": (
            "PASS"
            if all(
                c["valid_distribution"] and c["repeat_max_abs_drift"] == 0
                for c in cases
            )
            and padding_drift < 1e-4
            else "FAIL"
        ),
        "source": metadata["source"],
        "loaded_parameters": metadata["loaded_parameters"],
        "backbone_parameters": metadata["backbone_parameters"],
        "head_parameters": metadata["head_parameters"],
        "discarded_pretraining_head_keys": metadata["discarded_pretraining_head_keys"],
        "token_ids": metadata["token_ids"],
        "cases": cases,
        "padding_max_abs_drift": padding_drift,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=tuple(enc.SOURCES), required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Pinned Kai bundle for System One conversion",
    )
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
