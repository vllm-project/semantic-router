"""Collect pinned Bosun v3.1 0.6B native typed decisions on gold-free prompts.

Bosun's published custom loader and ``predict`` method own the prompt and
decision-token readout. This adapter only maps the shared Choice/Noul/Score
input and output contract, and records the exact source and base packages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .run import digest, file_digest, load_prompts, local_revision, synchronize

VARIANTS = {
    "0.6b": (
        "Hanno-Labs/bosun-v3.1-0.6b",
        "1d8b6f9611f9b64b514ce8b57cd86398fbc31a3b",
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "bosun-v31-06b-native-predict-v1",
    ),
    "1.7b": (
        "Hanno-Labs/bosun-v3.1-1.7b",
        "1d8dc82a20e4a32ed60927a47272d6efff48eed2",
        "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "bosun-v31-17b-native-predict-v1",
    ),
}
MODEL_ID, MODEL_REVISION, BASE_ID, BASE_REVISION, ADAPTER_VERSION = VARIANTS["0.6b"]


def _manifest_files(model_path: Path) -> dict[str, str]:
    manifest = json.loads((model_path / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Bosun package has no file manifest")
    for name, expected in files.items():
        relative = Path(name)
        if (
            not isinstance(name, str)
            or relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            raise ValueError("Malformed Bosun package manifest")
        file = model_path / relative
        if file.is_symlink() or not file.is_file() or file_digest(file) != expected:
            raise ValueError(f"Bosun package file differs: {name}")
    return files


def verify_packages(
    model_path: Path, base_path: Path, size: str = "0.6b"
) -> dict[str, Any]:
    if size not in VARIANTS:
        raise ValueError(f"Unknown Bosun size: {size}")
    model_id, model_revision, base_id, base_revision, _ = VARIANTS[size]
    if not local_revision(model_path, model_revision):
        raise ValueError("Bosun source revision is not locally attested")
    if not local_revision(base_path, base_revision):
        raise ValueError("Bosun's Qwen base revision is not locally attested")
    files = _manifest_files(model_path)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("base_model_name_or_path") != base_id
        or config.get("base_model_revision") != base_revision
        or config.get("prompt_schema") != "bosun-decision-prompt-v3-stable-slots"
        or config.get("decision_token_count") != 256
        or config.get("decision_token_assignment") != "presented_slot"
    ):
        raise ValueError("Bosun native loading or decision contract differs")
    base_files = [base_path / "config.json", *sorted(base_path.glob("*.safetensors"))]
    if len(base_files) < 2 or any(not file.is_file() for file in base_files):
        raise ValueError("Pinned Bosun base package is incomplete")
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "base_model_id": base_id,
        "base_revision": base_revision,
        "source_manifest_sha256": file_digest(model_path / "manifest.json"),
        "source_files_sha256": files,
        "base_files_sha256": {file.name: file_digest(file) for file in base_files},
    }


def candidates_for(question: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    kind = question.get("type")
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("Choice requires at least two criteria")
        pairs = [(str(key), str(value)) for key, value in criteria.items()]
        candidates = [
            {"id": key, "label": key, "description": description}
            for key, description in pairs
        ]
    elif kind == "noul":
        criteria = question.get("criteria")
        true_description = (
            str(criteria["true"])
            if isinstance(criteria, dict) and "true" in criteria
            else "The instruction is true."
        )
        false_description = (
            str(criteria["false"])
            if isinstance(criteria, dict) and "false" in criteria
            else "The instruction is false."
        )
        candidates = [
            {"id": "true", "label": "True", "description": true_description},
            {"id": "false", "label": "False", "description": false_description},
        ]
    elif kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("Score requires at least two ordered levels")
        candidates = [
            {"id": str(index), "label": str(value), "description": ""}
            for index, value in enumerate(criteria)
        ]
    else:
        raise ValueError(f"Unsupported Decision type: {kind!r}")
    if len(candidates) > 255:
        raise ValueError("Bosun supports at most 255 candidates")
    return candidates, [item["id"] for item in candidates]


def project(
    question: dict[str, Any], keys: list[str], values: list[float]
) -> dict[str, Any]:
    if (
        len(keys) != len(values)
        or any(not math.isfinite(p) or p < 0 or p > 1 for p in values)
        or abs(sum(values) - 1) > 1e-4
    ):
        raise ValueError("Bosun native probabilities are malformed")
    best = max(range(len(keys)), key=values.__getitem__)
    kind = question["type"]
    if kind == "noul":
        return {"type": kind, "noul": values[keys.index("true")]}
    probabilities = dict(zip(keys, values, strict=True))
    if kind == "choice":
        return {
            "type": kind,
            "probabilities": probabilities,
            "confidence": values[best],
            "choice": keys[best],
        }
    return {
        "type": kind,
        "probabilities": probabilities,
        "confidence": values[best],
        "score": sum(index * value for index, value in enumerate(values)),
        "legend": dict(zip(keys, question["criteria"], strict=True)),
    }


def collect(
    *,
    model_path: Path,
    base_path: Path,
    prompts: Path,
    output: Path,
    size: str = "0.6b",
) -> dict[str, Any]:
    if output.exists() or output.with_name(output.name + ".manifest.json").exists():
        raise FileExistsError("Refusing to overwrite Bosun predictions")
    rows = load_prompts(prompts)
    model_path, base_path = (
        model_path.resolve(strict=True),
        base_path.resolve(strict=True),
    )
    identity = verify_packages(model_path, base_path, size)
    model_id, model_revision, base_id, base_revision, adapter_version = VARIANTS[size]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.cuda.set_device(0)
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if (
        config.base_model_name_or_path != base_id
        or config.base_model_revision != base_revision
    ):
        raise ValueError("Loaded Bosun config differs from pinned manifest")
    # Only the lookup path changes; the base files and revision are checked above.
    config.base_model_name_or_path = str(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda:0")
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"items": 0, "questions": 0, "invalid_questions": 0}
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            answers: dict[str, dict[str, Any]] = {}
            synchronize("cuda:0")
            start = time.perf_counter()
            for question_id, question in row["questions"].items():
                counts["questions"] += 1
                try:
                    candidates, keys = candidates_for(question)
                    native = model.predict(
                        state=row["state"],
                        instructions=question["instructions"],
                        candidates=candidates,
                        decision_type=question["type"],
                        row_id=f"{row['id']}:{question_id}",
                        seed=0,
                    )
                    answers[question_id] = project(
                        question,
                        keys,
                        [float(value) for value in native["probabilities"]],
                    )
                except ValueError as error:
                    counts["invalid_questions"] += 1
                    answers[question_id] = {
                        "type": question["type"],
                        "error": type(error).__name__,
                    }
            synchronize("cuda:0")
            receipt = {
                "id": row["id"],
                "answers": answers,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "usage": None,
                "source_input_sha256": digest(
                    {"state": row["state"], "questions": row["questions"]}
                ),
                "backend": "bosun-v31-native",
                "model_id": model_id,
                "model_revision": model_revision,
                "adapter_version": adapter_version,
            }
            target.write(
                json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n"
            )
            target.flush()
            counts["items"] += 1
    manifest = {
        **identity,
        "adapter_version": adapter_version,
        "input_sha256": file_digest(prompts),
        "output_sha256": file_digest(output),
        "counts": counts,
        "loaded_parameter_count": parameter_count,
        "runtime": {
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "dtype": "bfloat16",
            "backend": "BosunForDecision.predict",
        },
    }
    output.with_name(output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=tuple(VARIANTS), default="0.6b")
    args = parser.parse_args()
    result = collect(
        model_path=args.model_path,
        base_path=args.base_path,
        prompts=args.input,
        output=args.output,
        size=args.size,
    )
    print(json.dumps({"output": str(args.output), "counts": result["counts"]}))


if __name__ == "__main__":
    main()
