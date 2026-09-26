"""Native This-That 1.0 collector for gold-free typed-decision prompts.

The released model has one generic declared-option head. Noul and Score are
explicit, documented projections of that head rather than native typed heads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from inference.run import digest, file_digest, load_prompts, local_revision, synchronize

MODEL_ID = "flock-io/this-that-model-1.0"
MODEL_REVISION = "3d927195c4f9845efe66c5715883a7a0f42b1239"
SOURCE_REVISION = "4efe782ccbb9c1979a9951c35a29a8e0b3b80bf0"
ARTIFACT_SHA256 = {
    "config.json": "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    "model.safetensors": "11bab4bbbce0214dcb4d70a88e74b3e4bde6fe8a95f239e3d0c355946e0000e0",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}
ADAPTER_VERSION = "this-that-native-v1"
MAX_STATE_TOKENS = 1536
MAX_CONTEXT_TOKENS = 262144


def verify_release(
    model_path: Path, source_path: Path, revision: str
) -> dict[str, Any]:
    if revision != MODEL_REVISION:
        raise ValueError(f"This-That 1.0 requires exact HF revision {MODEL_REVISION}")
    if not local_revision(model_path, revision):
        raise ValueError(
            "HF local-dir metadata is required to attest the exact revision"
        )
    for name, expected in ARTIFACT_SHA256.items():
        if file_digest(model_path / name) != expected:
            raise ValueError(f"Published This-That artifact hash mismatch: {name}")
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "qwen3_5_text"
        or config.get("architectures") != ["Qwen3_5ForCausalLM"]
        or config.get("max_position_embeddings") != MAX_CONTEXT_TOKENS
    ):
        raise ValueError("Unexpected This-That backbone/configuration")
    commit = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != SOURCE_REVISION:
        raise ValueError("This-That source does not match the 1.0 release commit")
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            str(source_path),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if changed:
        raise ValueError("This-That source has modified tracked files")
    if not (source_path / "thisthat/model.py").is_file():
        raise FileNotFoundError("Native This-That API is missing")
    return {
        "backend": "this-that",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "revision_attested": True,
        "source_revision": SOURCE_REVISION,
        "model_config_sha256": ARTIFACT_SHA256["config.json"],
        "model_weights_sha256": ARTIFACT_SHA256["model.safetensors"],
    }


def content(value: Any, field: str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{field}: empty text")
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    raise ValueError(f"{field}: expected text, object or array")


def native_questions(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Preserve each key and criterion while projecting to declared options."""
    state = content(row["state"], "state")
    specs = []
    for qid, question in row["questions"].items():
        if not isinstance(question, dict):
            raise ValueError(f"{row['id']}/{qid}: question must be an object")
        kind = question.get("type")
        instructions = content(
            question.get("instructions"), f"{row['id']}/{qid}/instructions"
        )
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError(f"{row['id']}/{qid}: Choice requires 2..255 criteria")
            labels = list(criteria)
            options = [
                (
                    label
                    if description is None
                    else label
                    + ": "
                    + content(description, f"{row['id']}/{qid}/{label}")
                )
                for label, description in criteria.items()
            ]
            projection = "native_declared_choice"
        elif kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
            ):
                raise ValueError(
                    f"{row['id']}/{qid}: Noul criteria need false/true only"
                )
            labels = ["no", "yes"]
            options = [
                (
                    label
                    if criteria is None or key not in criteria
                    else label
                    + ": "
                    + content(criteria[key], f"{row['id']}/{qid}/{key}")
                )
                for label, key in (("no", "false"), ("yes", "true"))
            ]
            projection = "binary_declared_choice"
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError(f"{row['id']}/{qid}: Score requires 2..10 levels")
            labels = [str(i) for i in range(len(criteria))]
            options = [
                f"{i}: " + content(level, f"{row['id']}/{qid}/{i}")
                for i, level in enumerate(criteria)
            ]
            projection = "ordered_declared_choice_expected_index"
        else:
            raise ValueError(f"{row['id']}/{qid}: unsupported question type {kind!r}")
        if len(set(options)) != len(options):
            raise ValueError(f"{row['id']}/{qid}: projected native options collide")
        specs.append(
            {
                "id": qid,
                "type": kind,
                "instructions": instructions,
                "labels": labels,
                "options": options,
                "projection": projection,
            }
        )
    return state, specs


def answer_from_native(decision: Any, spec: dict[str, Any]) -> dict[str, Any]:
    labels, options = spec["labels"], spec["options"]
    if tuple(decision.options) != tuple(options):
        raise ValueError("Native This-That option order differs from the request")
    probs = tuple(float(p) for p in decision.probabilities)
    if (
        len(probs) != len(labels)
        or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs)
        or abs(sum(probs) - 1) > 2e-5
        or type(decision.index) is not int
        or not 0 <= decision.index < len(labels)
        or decision.index != max(range(len(probs)), key=probs.__getitem__)
    ):
        raise ValueError("Native This-That distribution is invalid")
    kind = spec["type"]
    if kind == "noul":
        return {
            "type": kind,
            "noul": probs[1],
            "probabilities": {"no": probs[0], "yes": probs[1]},
            "projection": spec["projection"],
        }
    result = {
        "type": kind,
        "probabilities": dict(zip(labels, probs)),
        "confidence": probs[decision.index],
        "projection": spec["projection"],
    }
    if kind == "choice":
        result["choice"] = labels[decision.index]
    else:
        result["score"] = sum(i * p for i, p in enumerate(probs))
    return result


def completed_ids(
    output: Path, rows: list[dict[str, Any]], identity: dict[str, Any]
) -> set[str]:
    expected = {
        row["id"]: {
            "hash": digest({"state": row["state"], "questions": row["questions"]}),
            "question_ids": set(row["questions"]),
        }
        for row in rows
    }
    completed = set()
    with output.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            item = json.loads(line)
            item_id = item.get("id")
            if item_id not in expected or item_id in completed:
                raise ValueError(
                    f"{output}:{number}: unknown or duplicate prediction ID"
                )
            for key in (
                "backend",
                "model_id",
                "model_revision",
                "source_revision",
                "revision_attested",
                "model_config_sha256",
                "model_weights_sha256",
            ):
                if item.get(key) != identity[key]:
                    raise ValueError(f"{output}:{number}: stale model/source identity")
            if (
                item.get("adapter_version") != ADAPTER_VERSION
                or item.get("source_input_sha256") != expected[item_id]["hash"]
                or not isinstance(item.get("answers"), dict)
                or set(item["answers"]) != expected[item_id]["question_ids"]
            ):
                raise ValueError(f"{output}:{number}: stale input or adapter")
            completed.add(item_id)
    return completed


def load_native(model_path: Path, source_path: Path, device: str):
    if device != "cuda:0":
        raise ValueError("Expose one AMD ROCm GPU as cuda:0")
    loaded = sys.modules.get("thisthat")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if (
            loaded_file is None
            or Path(loaded_file).resolve().parent
            != (source_path / "thisthat").resolve()
        ):
            raise RuntimeError(
                "A different This-That package is loaded in this process"
            )
    sys.path.insert(0, str(source_path))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from thisthat import Question, TypedDecider
    from thisthat.prompt import build

    if (
        torch.version.hip is None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("This-That ROCm evaluation requires one visible AMD GPU")
    decider = TypedDecider.from_pretrained(
        str(model_path),
        device="cuda",
        local_files_only=True,
    )
    return (
        decider,
        Question,
        build,
        {
            "torch": str(torch.__version__),
            "hip": torch.version.hip,
            "native_dtype": "bfloat16",
            "native_temperature": 1.0,
            "native_layout": "state_first",
            "native_max_state_tokens": MAX_STATE_TOKENS,
            "runtime_qualification": "unvalidated_rocm",
        },
    )


def collect(
    *,
    model_path: Path,
    source_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    resume: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    rows = load_prompts(prompts)
    # Validate every supported question type before loading any GPU weights.
    for row in rows:
        native_questions(row)
    model_path = model_path.resolve(strict=True)
    source_path = source_path.resolve(strict=True)
    identity = verify_release(model_path, source_path, revision)
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"{output} exists; --resume requires the same input and model"
            )
        completed = completed_ids(output, rows, identity)
    else:
        completed = set()
    remaining = [row for row in rows if row["id"] not in completed]
    if max_items is not None:
        remaining = remaining[:max_items]
    if not remaining:
        return {
            **identity,
            "input_items": len(rows),
            "previously_completed": len(completed),
            "collected_now": 0,
            "output": str(output),
        }
    decider, Question, build, runtime = load_native(model_path, source_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    overflow_count = 0
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            state, specs = native_questions(row)
            questions = [
                Question(spec["instructions"], spec["options"]) for spec in specs
            ]
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            state_tokens = len(
                decider.tokenizer.encode(
                    "Context:\n" + state,
                    add_special_tokens=False,
                )
            )
            if state_tokens > MAX_STATE_TOKENS + 3:
                invalid_reason = "context_overflow"
                usage = None
                answers = {
                    spec["id"]: {
                        "type": spec["type"],
                        "error": invalid_reason,
                        "projection": spec["projection"],
                    }
                    for spec in specs
                }
                overflow_count += 1
            else:
                built = build(
                    decider.tokenizer,
                    state,
                    questions,
                    layout="state_first",
                    max_state_tokens=MAX_STATE_TOKENS,
                )
                if len(built["ids"]) > MAX_CONTEXT_TOKENS:
                    invalid_reason = "context_overflow"
                    usage = None
                    answers = {
                        spec["id"]: {
                            "type": spec["type"],
                            "error": invalid_reason,
                            "projection": spec["projection"],
                        }
                        for spec in specs
                    }
                    overflow_count += 1
                else:
                    invalid_reason = None
                    decisions = decider.decide(
                        state,
                        questions,
                        temperature=1.0,
                        layout="state_first",
                        max_state_tokens=MAX_STATE_TOKENS,
                    )
                    if len(decisions) != len(specs):
                        raise RuntimeError(f"{row['id']}: incomplete native decisions")
                    answers = {
                        spec["id"]: answer_from_native(decision, spec)
                        for spec, decision in zip(specs, decisions)
                    }
                    usage = {"input_tokens": len(built["ids"]), "output_tokens": 0}
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if not math.isfinite(latency_ms) or latency_ms < 0:
                raise ValueError(f"{row['id']}: invalid latency")
            receipt = {
                **identity,
                "id": row["id"],
                "answers": answers,
                "usage": usage,
                "latency_ms": latency_ms,
                "adapter_version": ADAPTER_VERSION,
                "source_input_sha256": digest(payload),
                "invalid_reason": invalid_reason,
                "native_output_contract": "generic_declared_options",
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
        **identity,
        **runtime,
        "input_items": len(rows),
        "previously_completed": len(completed),
        "collected_now": len(remaining),
        "context_overflow_now": overflow_count,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--input", type=Path, help="Gold-free benchmark prompts JSONL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--verify-only", action="store_true", help="CPU-only pinned source/model checks"
    )
    args = parser.parse_args()
    if args.verify_only:
        print(
            json.dumps(
                verify_release(
                    args.model_path.resolve(strict=True),
                    args.source_path.resolve(strict=True),
                    args.model_revision,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --verify-only is set")
    result = collect(
        model_path=args.model_path,
        source_path=args.source_path,
        revision=args.model_revision,
        prompts=args.input,
        output=args.output,
        device=args.device,
        resume=args.resume,
        max_items=args.max_items,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
