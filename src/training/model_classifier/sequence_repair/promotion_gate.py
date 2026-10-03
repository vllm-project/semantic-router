import argparse
import hashlib
import json
from pathlib import Path


def read_predictions(evaluation_path):
    path = Path(evaluation_path).with_suffix(".predictions.jsonl")
    return {
        row["id"]: row["prediction"]
        for row in map(json.loads, path.read_text().splitlines())
    }


def decide(
    gate, candidate, baseline, candidate_predictions, baseline_predictions, conformance
):
    if candidate["data_files"] != baseline["data_files"]:
        raise ValueError("Candidate and baseline were evaluated on different data")
    if candidate_predictions.keys() != baseline_predictions.keys():
        raise ValueError("Candidate and baseline predictions cover different rows")
    rules = gate["promote_if"]
    macro_delta = candidate["macro_f1"] - baseline["macro_f1"]
    language_deltas = {
        language: candidate["breakdowns"]["language"][language]["macro_f1"]
        - metrics["macro_f1"]
        for language, metrics in sorted(baseline["breakdowns"]["language"].items())
    }
    failures = ["macro_f1"] if macro_delta < rules["macro_f1_min_delta"] else []
    failures += [
        f"language:{language}"
        for language, delta in language_deltas.items()
        if delta < rules["per_language_macro_f1_min_delta"]
    ]
    failures += [
        f"conformance:{check['name']}"
        for check in conformance["checks"]
        if not check["passed"]
    ]
    agreement = sum(
        candidate_predictions[key] == baseline_predictions[key]
        for key in baseline_predictions
    ) / len(baseline_predictions)
    return {
        "decision": "reject" if failures else "promote",
        "failures": failures,
        "candidate_macro_f1": candidate["macro_f1"],
        "baseline_macro_f1": baseline["macro_f1"],
        "macro_f1_delta": macro_delta,
        "language_macro_f1_delta": language_deltas,
        "prediction_agreement": agreement,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--conformance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a promotion decision")
    receipt = decide(
        json.loads(args.gate.read_text()),
        json.loads(args.candidate.read_text()),
        json.loads(args.baseline.read_text()),
        read_predictions(args.candidate),
        read_predictions(args.baseline),
        json.loads(args.conformance.read_text()),
    )
    receipt["inputs_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [
            ("gate", args.gate),
            ("candidate", args.candidate),
            ("baseline", args.baseline),
            ("conformance", args.conformance),
        ]
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
