"""Bounded gold-free probe for Eikos cross-process native readout drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.eikos import shared_answer
from inference.run import load_prompts
from training.eikos.io import atomic_json
from training.eikos.native import load_decider
from training.eikos.verify_export import compare_answers, verify_sums
from training.model.data import file_sha256


def load_answers(path: Path) -> dict[str, dict]:
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in result or len(row["answers"]) != 1:
                raise ValueError("Reference predictions need unique one-question rows")
            result[row["id"]] = row
    return result


def probe(
    *,
    model_path: Path,
    prompts: Path,
    reference_a: Path,
    reference_b: Path,
    output: Path,
    max_items: int = 20,
    repeats: int = 3,
    deterministic: bool = False,
    device: str = "cuda:0",
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= max_items <= 50 or not 2 <= repeats <= 5:
        raise ValueError("Probe must stay bounded to <=50 rows and <=5 repeats")
    verify_sums(model_path)
    a, b = load_answers(reference_a), load_answers(reference_b)
    items = {row["id"]: row for row in load_prompts(prompts)}
    if set(a) != set(b) or set(a) != set(items):
        raise ValueError("Probe reference IDs differ from panel")
    ordered = []
    for item_id in items:
        left = next(iter(a[item_id]["answers"].values()))
        right = next(iter(b[item_id]["answers"].values()))
        changed, drift, _ = compare_answers(left, right)
        ordered.append((changed, drift, item_id))
    selected = sorted(ordered, key=lambda row: (-int(row[0]), -row[1], row[2]))[
        :max_items
    ]
    import torch

    if deterministic:
        torch.use_deterministic_algorithms(True)
    native = load_decider(model_path, None, model_path / "calib.json", device=device)
    observations = []
    for prior_changed, prior_drift, item_id in selected:
        row = items[item_id]
        key = next(iter(row["questions"]))
        answers, tokens = [], []
        for _ in range(repeats):
            result = native.decide_all(row["state"], row["questions"])
            answer, n_tok = result[key]
            answers.append(shared_answer(row["questions"][key], answer))
            tokens.append(n_tok)
        if len(set(tokens)) != 1:
            raise ValueError("Repeated in-process token count changed")
        within = [compare_answers(answers[0], answer) for answer in answers[1:]]
        ref_a = next(iter(a[item_id]["answers"].values()))
        ref_b = next(iter(b[item_id]["answers"].values()))
        drift_a = compare_answers(ref_a, answers[0])[1]
        drift_b = compare_answers(ref_b, answers[0])[1]
        observations.append(
            {
                "id": item_id,
                "input_tokens": tokens[0],
                "prior_cross_process_categorical_change": prior_changed,
                "prior_cross_process_probability_drift": prior_drift,
                "within_process_categorical_change": any(
                    not same for same, _, _ in within
                ),
                "within_process_max_probability_drift": max(
                    drift for _, drift, _ in within
                ),
                "first_output_probability_drift_from_ref_a": drift_a,
                "first_output_probability_drift_from_ref_b": drift_b,
            }
        )
    report = {
        "schema_version": "decision2-eikos-determinism-probe/1",
        "scope": "gold-free diagnostic of preselected CSS pilot disagreements; no model selection",
        "model_sha256": file_sha256(model_path / "SHA256SUMS"),
        "calibration_sha256": file_sha256(model_path / "calib.json"),
        "prompts_sha256": file_sha256(prompts),
        "reference_a_sha256": file_sha256(reference_a),
        "reference_b_sha256": file_sha256(reference_b),
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "items": len(observations),
        "repeats": repeats,
        "within_process_categorical_changes": sum(
            row["within_process_categorical_change"] for row in observations
        ),
        "within_process_max_probability_drift": max(
            row["within_process_max_probability_drift"] for row in observations
        ),
        "observations": observations,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--reference-a", type=Path, required=True)
    parser.add_argument("--reference-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = probe(
        model_path=args.model_path,
        prompts=args.prompts,
        reference_a=args.reference_a,
        reference_b=args.reference_b,
        output=args.output,
        max_items=args.max_items,
        repeats=args.repeats,
        deterministic=args.deterministic,
        device=args.device,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "items",
                    "repeats",
                    "torch_deterministic_algorithms",
                    "within_process_categorical_changes",
                    "within_process_max_probability_drift",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
