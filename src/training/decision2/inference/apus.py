"""Evaluate pinned APUS OpenJev merged releases through their native OpenJet API.

APUS's portable runtime admits dynamic Choice and binary Noul. It does not
implement a complete ordinal Score question, which remains an explicit
ineligible answer in the common benchmark denominator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .run import digest, file_digest, load_prompts, local_revision, synchronize

MODELS = {
    "4b": (
        "apus-ailab/APUS-OpenJev-v1-4B",
        "422b3741f8b5c092eeefef847c1ca89d78337d45",
        5949,
    ),
    "9b": (
        "apus-ailab/APUS-OpenJev-v1-9B",
        "82c9c56cfa9de8d36704ed91948d4726ef111635",
        3000,
    ),
}
ADAPTER_VERSION = "apus-native-openjet-v1"
# The two cards were edited after their immutable release manifests. These
# documents alone are exempt from the manifest's old hashes; weights, native
# code, tokenizer, chat template and configs must match exactly.
UPDATED_CARD_FILES = frozenset({"README.md", "README.zh-CN.md"})


def verify_release(path: Path, size: str, revision: str) -> dict[str, Any]:
    model_id, required_revision, step = MODELS[size]
    if revision != required_revision or not local_revision(path, revision):
        raise ValueError(f"{model_id}: local files lack the pinned HF revision")
    manifest_path = path / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "openjet.merged-release.v1"
        or manifest.get("size") != size.upper()
        or manifest.get("checkpoint_step") != step
        or manifest.get("repo_id") != model_id
    ):
        raise ValueError("Unexpected APUS release identity")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Missing APUS release file roster")
    seen: set[str] = set()
    for entry in files:
        name = entry.get("path") if isinstance(entry, dict) else None
        if (
            not isinstance(name, str)
            or name in seen
            or name.startswith("/")
            or "\\" in name
            or ".." in Path(name).parts
        ):
            raise ValueError("Unsafe or duplicate APUS release path")
        seen.add(name)
        item = path / name
        if (
            not item.is_file()
            or item.is_symlink()
            or not item.resolve().is_relative_to(path.resolve())
        ):
            raise ValueError(f"Missing APUS release file: {name}")
        if name in UPDATED_CARD_FILES:
            continue
        if item.stat().st_size != entry.get("bytes") or file_digest(item) != entry.get(
            "sha256"
        ):
            raise ValueError(f"APUS release file differs from manifest: {name}")
    required = {
        "config.json",
        "depth_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "openjet_runtime/runtime.py",
        "openjet_runtime/contracts.py",
        "openjet_runtime/early_exit.py",
        "openjet_runtime/candidate_projection.py",
    }
    if not required.issubset(seen) or not any(
        name.endswith(".safetensors") for name in seen
    ):
        raise ValueError("Incomplete APUS native runtime or weights")
    depth = json.loads((path / "depth_config.json").read_text(encoding="utf-8"))
    if (
        depth.get("prompt_version") != "jev.dynamic.prompt.v2"
        or depth.get("exit_depth") != 16
        or depth.get("full_depth") != 32
        or depth.get("checkpoint_step") != step
    ):
        raise ValueError("Unexpected APUS native prompt or exit-depth contract")
    return {
        "release_manifest_sha256": file_digest(manifest_path),
        "native_runtime_sha256": file_digest(path / "openjet_runtime/runtime.py"),
        "model_config_sha256": file_digest(path / "config.json"),
        "card_sha256": file_digest(path / "README.md"),
        "verified_release_files": len(seen) - len(UPDATED_CARD_FILES),
    }


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def native_request(row: dict[str, Any], question_id: str) -> dict[str, Any] | None:
    question = row["questions"][question_id]
    qtype = question["type"]
    if qtype == "score":
        return None
    instructions = question["instructions"]
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Question instructions must be nonempty")
    if qtype == "choice":
        criteria = question["criteria"]
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 16:
            raise ValueError("APUS Choice requires 2..16 candidates")
        # The native prompt renderer hides candidate IDs and presents only
        # descriptions; include the original IDs so tasks referring to record
        # names remain answerable. Preserve original candidate order.
        candidates = [
            {"id": label, "description": f"{label}: {description}"}
            for label, description in criteria.items()
        ]
        primitive = "choice"
    elif qtype == "noul":
        criteria = question["criteria"]
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise ValueError("APUS Noul requires true/false criteria")
        instructions += (
            f"\nYes means: {criteria['true']}" f"\nNo means: {criteria['false']}"
        )
        candidates = [
            {"id": "yes", "description": "The stated proposition is true."},
            {"id": "no", "description": "The stated proposition is false."},
        ]
        primitive = "noul"
    else:
        raise ValueError(f"Unsupported question type: {qtype}")
    return {
        "id": f"{row['id']}/{question_id}",
        "group_id": row["id"],
        "state": state_text(row["state"]),
        "instructions": instructions,
        "primitive": primitive,
        "criteria": candidates,
    }


def shared_answer(qtype: str, native: dict[str, Any]) -> dict[str, Any]:
    if qtype == "choice":
        probabilities = native.get("probabilities")
        label = native.get("choice")
        if (
            not isinstance(probabilities, dict)
            or label not in probabilities
            or not all(
                type(p) in (int, float) and math.isfinite(p)
                for p in probabilities.values()
            )
        ):
            raise ValueError("Malformed APUS Choice output")
        return {
            "type": "choice",
            "choice": label,
            "probabilities": probabilities,
            "native_effort": native.get("effort"),
            "native_projection": native.get("projection"),
            "calibrated": native.get("calibrated"),
        }
    if qtype == "noul":
        probability = native.get("yes_probability")
        if type(probability) not in (int, float) or not math.isfinite(probability):
            raise ValueError("Malformed APUS Noul output")
        return {
            "type": "noul",
            "noul": float(probability),
            "native_effort": native.get("effort"),
            "native_projection": native.get("projection"),
            "calibrated": native.get("calibrated"),
        }
    raise ValueError(f"Unsupported APUS question type: {qtype}")


def completed_rows(
    path: Path, rows: list[dict[str, Any]], identity: dict[str, Any]
) -> set[str]:
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in rows
    }
    completed: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        receipt = json.loads(line)
        item_id = receipt.get("id")
        if item_id not in expected or item_id in completed:
            raise ValueError(f"{path}:{number}: unexpected or duplicate APUS ID")
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError(f"{path}:{number}: stale APUS identity")
        if (
            receipt.get("source_input_sha256") != expected[item_id][0]
            or not isinstance(receipt.get("answers"), dict)
            or set(receipt["answers"]) != expected[item_id][1]
        ):
            raise ValueError(f"{path}:{number}: stale or incomplete APUS answers")
        completed.add(item_id)
    return completed


def load_native(path: Path, device: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    import transformers

    if transformers.__version__ != "5.16.1":
        raise RuntimeError("APUS native runtime requires Transformers 5.16.1")
    sys.path.insert(0, str(path))
    from openjet_runtime import OpenJet

    return OpenJet.from_pretrained(str(path), device=device, dtype="bfloat16")


def collect(
    *,
    size: str,
    model_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    effort: str = "high",
    resume: bool = False,
    max_items: int | None = None,
    verify_only: bool = False,
) -> dict[str, Any]:
    if size not in MODELS or effort not in ("low", "high"):
        raise ValueError("Expected APUS size 4b/9b and effort low/high")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    path = model_path.resolve(strict=True)
    release = verify_release(path, size, revision)
    model_id, _, _ = MODELS[size]
    identity = {
        "backend": "apus",
        "model_id": model_id,
        "model_revision": revision,
        "revision_attested": True,
        "adapter_version": ADAPTER_VERSION,
        "effort": effort,
        **release,
    }
    if verify_only:
        return identity
    rows = load_prompts(prompts)
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        completed = completed_rows(output, rows, identity)
    else:
        completed = set()
    remaining = [row for row in rows if row["id"] not in completed]
    if max_items is not None:
        remaining = remaining[:max_items]
    native = load_native(path, device)
    import torch

    runtime = {
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "transformers": version("transformers"),
        "runtime_qualification": "unvalidated_rocm_apus_native",
    }
    counts: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            answers: dict[str, Any] = {}
            input_tokens = 0
            synchronize(device)
            started = time.perf_counter()
            for question_id, question in row["questions"].items():
                request = native_request(row, question_id)
                if request is None:
                    reason = "unsupported_native_ordinal_score"
                    answers[question_id] = {"type": question["type"], "error": reason}
                    counts[reason] += 1
                    continue
                try:
                    response = native.decide(request, effort=effort)
                except ValueError as error:
                    message = str(error)
                    if "input exceeds runtime limit" in message:
                        reason = "context_overflow"
                    elif (
                        "candidate label" in message
                        or "multimodal placeholders" in message
                    ):
                        reason = "native_input_ineligible"
                    else:
                        raise
                    answers[question_id] = {"type": question["type"], "error": reason}
                    counts[reason] += 1
                    continue
                answer = shared_answer(question["type"], response)
                if answer.get("calibrated") is not False:
                    raise ValueError(
                        "APUS runtime unexpectedly changed calibration contract"
                    )
                answers[question_id] = answer
                input_tokens += int(response["prompt_tokens"])
                counts["valid_native_response"] += 1
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if set(answers) != set(row["questions"]) or not math.isfinite(latency_ms):
                raise ValueError(f"{row['id']}: malformed APUS native response")
            receipt = {
                "id": row["id"],
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                "latency_ms": latency_ms,
                "source_input_sha256": digest(payload),
                "model": f"{model_id}@{revision}",
                **identity,
                **runtime,
            }
            target.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            target.flush()
    return {
        "input_items": len(rows),
        "previously_completed": len(completed),
        "collected_now": len(remaining),
        "question_counts_collected_now": dict(counts),
        "output": str(output),
        **identity,
        **runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(MODELS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--effort", choices=("low", "high"), default="high")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and (args.input is None or args.output is None):
        parser.error("--input and --output are required for inference")
    print(
        json.dumps(
            collect(
                size=args.size,
                model_path=args.model_path,
                revision=args.model_revision,
                prompts=args.input,
                output=args.output,
                device=args.device,
                effort=args.effort,
                resume=args.resume,
                max_items=args.max_items,
                verify_only=args.verify_only,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
