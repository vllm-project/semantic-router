"""Gold-free native PEFT package-versus-scored-source BF16 parity gate.

Run only on the authorized model-evaluation GPU environment. The receipt
contains hashes and aggregate differences, never prompts or answers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

from training.model.data import canonical, file_sha256
from training.model.infer import checkpoint_fingerprint, load_prompts, run_prompts

from .adapter_runtime import _hash

VERSION = "decision2-peft-native-parity/1"
MAX_DRIFT = 1e-4
TYPES = {"choice", "noul", "score"}


def _answers_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical(rows).encode("utf-8")).hexdigest()


def _native_values(
    question: dict[str, Any], answer: dict[str, Any]
) -> tuple[list[float], str | bool | None] | None:
    kind = question["type"]
    if kind == "noul":
        probability = answer.get("noul")
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            return None
        point = None if probability == 0.5 else probability > 0.5
        return [float(probability)], point

    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict):
        keys = list(criteria)
    elif kind == "score" and isinstance(criteria, list):
        keys = [str(index) for index in range(len(criteria))]
    else:
        return None
    probabilities = answer.get("probabilities")
    if (
        len(keys) < 2
        or not isinstance(probabilities, dict)
        or set(probabilities) != set(keys)
    ):
        return None
    values = [probabilities[key] for key in keys]
    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in values
    ) or not math.isclose(sum(values), 1.0, abs_tol=1e-5):
        return None
    if kind == "choice":
        chosen = answer.get("choice")
        maximum = max(values)
        winners = [
            key for key, value in zip(keys, values) if abs(value - maximum) <= 1e-8
        ]
        if not isinstance(chosen, str) or len(winners) != 1 or chosen != winners[0]:
            return None
        return [float(value) for value in values], chosen
    score = answer.get("score")
    expected = sum(index * value for index, value in enumerate(values))
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not math.isclose(score, expected, abs_tol=1e-5)
    ):
        return None
    maximum = max(values)
    winners = [key for key, value in zip(keys, values) if abs(value - maximum) <= 1e-8]
    point = winners[0] if len(winners) == 1 else None
    return [*map(float, values), float(score)], point


def compare_answers(
    prompts: list[dict[str, Any]],
    source_predictions: list[dict[str, Any]],
    package_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed on absent, invalid, misordered or numerically different output."""
    if len(prompts) != len(source_predictions) or len(prompts) != len(
        package_predictions
    ):
        raise ValueError("Parity output count differs from gold-free roster")
    kinds: set[str] = set()
    mismatch = invalid = questions = 0
    largest = 0.0
    for prompt, left, right in zip(prompts, source_predictions, package_predictions):
        if left.get("id") != prompt["id"] or right.get("id") != prompt["id"]:
            raise ValueError("Parity output order or ID differs from roster")
        expected = prompt["questions"]
        la, ra = left.get("answers"), right.get("answers")
        if (
            not isinstance(la, dict)
            or not isinstance(ra, dict)
            or set(la) != set(expected)
            or set(ra) != set(expected)
        ):
            raise ValueError("Parity output lacks a complete question mapping")
        for qid, question in expected.items():
            kind = question.get("type")
            if kind not in TYPES:
                raise ValueError("Parity roster contains an unsupported question type")
            kinds.add(kind)
            questions += 1
            a, b = la[qid], ra[qid]
            if (
                not isinstance(a, dict)
                or not isinstance(b, dict)
                or a.get("type") != kind
                or b.get("type") != kind
            ):
                invalid += 1
                continue
            if "error" in a or "error" in b:
                invalid += 1
                continue
            av, bv = _native_values(question, a), _native_values(question, b)
            if av is None or bv is None:
                invalid += 1
                continue
            values = zip(av[0], bv[0], strict=True)
            for left_value, right_value in values:
                largest = max(largest, abs(left_value - right_value))
            if av[1] != bv[1]:
                mismatch += 1
    if kinds != TYPES:
        raise ValueError("Parity roster must contain native Choice, Noul and Score")
    return {
        "items": len(prompts),
        "questions": questions,
        "types": sorted(kinds),
        "invalid_or_missing_n": invalid,
        "categorical_mismatch_n": mismatch,
        "max_probability_or_score_drift": largest,
        "passed": invalid == 0 and mismatch == 0 and largest <= MAX_DRIFT,
        "predeclared_gate": {
            "invalid_or_missing_n": 0,
            "categorical_mismatch_n": 0,
            "max_probability_or_score_drift": MAX_DRIFT,
        },
    }


def _packaged_api(package: Path) -> Any:
    """Import the copied package code, not the source tree's runtime module."""
    init = package / "decision2/__init__.py"
    spec = importlib.util.spec_from_file_location(
        "decision2", init, submodule_search_locations=[str(init.parent)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the packaged Decision 2.0 loader")
    if any(
        name == "decision2" or name.startswith("decision2.") for name in sys.modules
    ):
        raise RuntimeError("Another decision2 runtime is already imported")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules["decision2"] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def run(
    *,
    package: Path,
    checkpoint: Path,
    source: Path,
    calibration: Path,
    scored_manifest: Path,
    prompts_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    package = package.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    source = source.resolve(strict=True)
    calibration = calibration.resolve(strict=True)
    scored_manifest = scored_manifest.resolve(strict=True)
    prompts_path = prompts_path.resolve(strict=True)
    api = _packaged_api(package)
    manifest = api.verify_bundle(package, source)
    api.api._dependencies(manifest)
    scored = json.loads(scored_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(scored, dict)
        or _hash(scored_manifest) != manifest["scored_prediction_manifest_sha256"]
        or scored.get("model_sha256") != manifest["model_sha256"]
        or _hash(calibration) != manifest["calibration_sha256"]
        or checkpoint_fingerprint(checkpoint, source)["model_sha256"]
        != manifest["model_sha256"]
    ):
        raise ValueError("Scored source and package identity differ")
    rows = load_prompts(prompts_path)

    import torch
    from training.model.decision_model import DecisionModel, collate, encode

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 parity requires the scored GPU runtime")
    device = torch.device("cuda:0")
    source_model, tokenizer = DecisionModel.from_checkpoint(
        checkpoint, source_path=source
    )
    source_model = source_model.float().to(device).eval()
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad is None:
        raise ValueError("Tokenizer needs pad or EOS")

    def predict(encoded: list[dict[str, Any]]) -> list[list[float]]:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in collate(encoded, pad).items()
        }
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = source_model(**batch)
        return [
            values[: len(item["keys"])].float().cpu().tolist()
            for values, item in zip(logits, encoded)
        ]

    left, counts = run_prompts(
        rows,
        tokenizer=tokenizer,
        max_length=manifest["max_length"],
        temperature=manifest["temperature_by_type"],
        encode_fn=encode,
        predict_fn=predict,
        model_sha256=manifest["model_sha256"],
        adapter_sha256=scored["adapter_sha256"],
        calibration_sha256=manifest["calibration_sha256"],
    )
    source_model = None
    tokenizer = None
    torch.cuda.empty_cache()
    packaged = api.Decision2.from_pretrained(
        package, source_path=source, device="cuda:0"
    )
    right = [
        {
            "id": row["id"],
            "answers": packaged.system_one(
                state=row["state"], questions=row["questions"]
            )["answers"],
        }
        for row in rows
    ]
    result = compare_answers(rows, left, right)
    receipt = {
        "schema_version": VERSION,
        "package_manifest_sha256": _hash(package / "MODEL_MANIFEST.json"),
        "scored_prediction_manifest_sha256": _hash(scored_manifest),
        "scored_predictions_sha256": scored["predictions_sha256"],
        "model_sha256": manifest["model_sha256"],
        "calibration_sha256": manifest["calibration_sha256"],
        "gold_free_prompts_sha256": file_sha256(prompts_path),
        "source_answers_sha256": _answers_digest([row["answers"] for row in left]),
        "package_answers_sha256": _answers_digest([row["answers"] for row in right]),
        "native_runtime": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device),
            "bf16": True,
            "one_item_batch": True,
            "no_truncation": True,
        },
        "source_counts": counts,
        **result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(
            receipt,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "package",
        "checkpoint",
        "source",
        "calibration",
        "scored_manifest",
        "prompts_path",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    receipt = run(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "passed": receipt["passed"],
                "items": receipt["items"],
                "questions": receipt["questions"],
                "max_drift": receipt["max_probability_or_score_drift"],
            }
        )
    )
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
