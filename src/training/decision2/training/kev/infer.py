"""Evaluate a locally trained, audited Kev-derived 4B checkpoint natively.

This is deliberately separate from the published Kev baseline collector: a
derived checkpoint must never inherit the published model's identity or score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inference.kev import KEV_BASE_ID, KEV_BASE_REVISION, KEV_SOURCE_REVISION
from inference.run import digest, file_digest, load_prompts, synchronize

from training.kev import prepare as converter
from training.kev.prepare import FORMAT, verify_parent
from training.model.data import canonical

ADAPTER_VERSION = "decision2-kev-derived-native/1"


def _content_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def verify_checkpoint(
    checkpoint: Path,
    parent_model: Path,
    source_path: Path,
    train_data: Path,
    train_manifest: Path,
    *,
    parent_check: Callable[[Path, Path], dict[str, Any]] = verify_parent,
) -> dict[str, Any]:
    """Bind weights to the converted TRAIN bytes and the published Kev parent."""
    parent = parent_check(parent_model, source_path)
    manifest = json.loads(train_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("Unknown Kev conversion manifest")
    if manifest.get("parent") != parent or manifest.get(
        "kev_train_sha256"
    ) != file_digest(train_data):
        raise ValueError("Kev conversion manifest does not match parent or TRAIN data")
    if manifest.get("conversion_code_sha256") != file_digest(Path(converter.__file__)):
        raise ValueError(
            "Kev conversion code differs from the audited training manifest"
        )
    preflight = manifest.get("native_preflight")
    rows = manifest.get("rows")
    if (
        not isinstance(preflight, dict)
        or type(rows) is not int
        or rows < 1
        or preflight.get("records") != rows
        or preflight.get("questions") != rows
        or preflight.get("native_source_revision") != KEV_SOURCE_REVISION
    ):
        raise ValueError("Kev conversion manifest lacks complete native preflight")
    config = json.loads(
        (checkpoint / "training_config.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (checkpoint / "training_metrics.json").read_text(encoding="utf-8")
    )
    adapter_config = json.loads(
        (checkpoint / "adapter_config.json").read_text(encoding="utf-8")
    )
    if any(not isinstance(item, dict) for item in (config, metrics, adapter_config)):
        raise ValueError("Native Kev checkpoint metadata must be JSON objects")
    args, init = config.get("args"), config.get("init_source")
    if not isinstance(args, dict) or not isinstance(init, dict):
        raise ValueError("Derived checkpoint lacks native warm-start provenance")
    required = {
        "base": KEV_BASE_ID,
        "base_revision": KEV_BASE_REVISION,
        "lora": 16,
        "head_dim": 256,
        "option_isolation": 0,
        "special_embeddings": 0,
        "weights_dtype": "fp32",
        "lora_placement": "full",
        "replay": 0,
        "suite": None,
        "lora_targets": "all",
        "public_frac": 1.0,
        "synthetic_repeat": 1,
        "train_sources": "",
        "anchor": "",
        "anchor_w": 0.0,
        "p_none": 0.0,
        "p_none_distract": 0.0,
        "p_distract": 0.0,
        "p_none_pair": 0.0,
        "max_state": preflight["training_context"]["max_state"],
    }
    for key, expected in required.items():
        if args.get(key) != expected:
            raise ValueError(
                f"Derived Kev run changed audited {key}: {args.get(key)!r}"
            )
    if (
        config.get("base_revision") != KEV_BASE_REVISION
        or init.get("adapter_sha256") != parent["adapter_sha256"]
        or init.get("head_sha256") != parent["head_sha256"]
    ):
        raise ValueError(
            "Derived checkpoint did not warm-start exact published adapter and head"
        )
    if (
        not isinstance(args.get("data"), str)
        or Path(args["data"]).resolve() != train_data.resolve()
    ):
        raise ValueError("Native trainer did not use this converted TRAIN file")
    if type(args.get("epochs")) is not int or args["epochs"] < 1:
        raise ValueError("Native trainer epochs are invalid")
    if (
        metrics.get("requested_records") != rows * args["epochs"]
        or metrics.get("truncated_records") != 0
        or metrics.get("rejected_records") != 0
    ):
        raise ValueError("Native trainer dropped or truncated audited TRAIN records")
    if (
        adapter_config.get("base_model_name_or_path") != KEV_BASE_ID
        or adapter_config.get("r") != 16
        or adapter_config.get("lora_alpha") != 32
        or adapter_config.get("lora_dropout") != 0.05
    ):
        raise ValueError(
            "Derived adapter architecture differs from the pinned Kev parent"
        )
    files = {
        name: file_digest(checkpoint / name)
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "head.pt",
            "training_config.json",
            "training_metrics.json",
            "tokenizer.json",
        )
    }
    identity = {
        "checkpoint_files_sha256": files,
        "parent_adapter_sha256": parent["adapter_sha256"],
        "parent_head_sha256": parent["head_sha256"],
        "parent_revision": parent["model_revision"],
        "source_revision": KEV_SOURCE_REVISION,
        "converted_train_sha256": manifest["kev_train_sha256"],
        "train_manifest_sha256": file_digest(train_manifest),
    }
    return {
        "model_sha256": _content_digest(identity),
        "identity": identity,
        "parent": parent,
        "rows": rows,
        "epochs": args["epochs"],
        "native_metrics": metrics,
    }


def load_native(checkpoint: Path, source_path: Path, device: str):
    """Use Kev's own Checkpoint, encoder, pointer head and answer formatter."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    loaded = sys.modules.get("kev")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded is not None and (
        loaded_file is None
        or Path(loaded_file).resolve().parent != (source_path / "kev").resolve()
    ):
        raise RuntimeError("A different Kev package is already imported")
    sys.path.insert(0, str(source_path))
    from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, ContextOverflow

    native = Checkpoint(str(checkpoint))
    if (
        native.meta.base != KEV_BASE_ID
        or native.meta.base_revision != KEV_BASE_REVISION
        or native.meta.head_dim != 256
        or native.meta.lora != 16
    ):
        raise ValueError("Derived Kev checkpoint metadata has the wrong architecture")
    tokenizer, model = native.load(device, LoadOptions())

    def decide(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        request = SystemOneRequest(
            state=state, questions=questions, model="decision2-kev-derived"
        )
        record, metadata = to_record(request)
        try:
            encoded = model.encode(
                tokenizer,
                record,
                max_state=SERVE_MAX_STATE,
                max_branch=SERVE_MAX_BRANCH,
                strict=True,
            )
        except ContextOverflow:
            return {
                "answers": {
                    name: {"type": spec["type"], "invalid_reason": "context_overflow"}
                    for name, spec in questions.items()
                },
                "status": "context_overflow",
                "usage": None,
            }
        probabilities = model.probs(encoded)
        if len(probabilities) != len(metadata):
            raise ValueError("Derived Kev model returned wrong question count")
        answers = to_answers([values.tolist() for values in probabilities], metadata)
        return {
            "answers": answers,
            "status": "ok",
            "usage": {
                "input_tokens": len(encoded["ids"]),
                "output_tokens": output_tokens(tokenizer, answers),
            },
        }

    return (
        model,
        decide,
        {
            "temperature": native.meta.temperature,
            "inference_dtype": "float32",
            "path": "Checkpoint.load/DecisionModel.probs/api.to_answers",
        },
    )


def _resume_ids(
    path: Path,
    prompts: list[dict[str, Any]],
    model_sha256: str,
    model_id: str,
    adapter_sha256: str,
) -> set[str]:
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in prompts
    }
    done: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            item_id = row.get("id")
            if item_id not in expected or item_id in done:
                raise ValueError(
                    f"{path}:{line_number}: unknown or duplicate prediction ID"
                )
            input_sha, question_ids = expected[item_id]
            if (
                row.get("source_input_sha256") != input_sha
                or row.get("model_sha256") != model_sha256
                or row.get("model_id") != model_id
                or row.get("adapter_sha256") != adapter_sha256
                or row.get("adapter_version") != ADAPTER_VERSION
                or not isinstance(row.get("answers"), dict)
                or set(row["answers"]) != question_ids
            ):
                raise ValueError(
                    f"{path}:{line_number}: stale model, input, or answer set"
                )
            done.add(item_id)
    return done


def collect(
    prompts_path: Path,
    output: Path,
    verification: dict[str, Any],
    *,
    model_id: str,
    decide: Callable[[Any, dict[str, Any]], dict[str, Any]],
    device: str = "cpu",
    resume: bool = False,
    max_items: int | None = None,
    runtime: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sync: Callable[[str], None] = synchronize,
) -> dict[str, Any]:
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    prompts = load_prompts(prompts_path)
    source_sha = file_digest(prompts_path)
    model_sha = verification["model_sha256"]
    adapter_sha = file_digest(Path(__file__))
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        completed = _resume_ids(output, prompts, model_sha, model_id, adapter_sha)
    else:
        completed = set()
    remaining = [row for row in prompts if row["id"] not in completed]
    if max_items is not None:
        remaining = remaining[:max_items]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if output.exists() else "x", encoding="utf-8") as stream:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            sync(device)
            start = clock()
            response = decide(**payload)
            sync(device)
            latency = (clock() - start) * 1000
            if (
                not isinstance(response, dict)
                or not isinstance(response.get("answers"), dict)
                or set(response["answers"]) != set(row["questions"])
            ):
                raise ValueError(
                    f"{row['id']}: native Kev answers do not match question IDs"
                )
            prediction = {
                "id": row["id"],
                "answers": response["answers"],
                "latency_ms": latency,
                "usage": response.get("usage"),
                "status": response.get("status", "ok"),
                "backend": "decision2-kev-derived",
                "model_id": model_id,
                "model_sha256": model_sha,
                "model_revision": "sha256:" + model_sha,
                "adapter_version": ADAPTER_VERSION,
                "adapter_sha256": adapter_sha,
                "input_sha256": digest(payload),
                "source_input_sha256": digest(payload),
                "source_file_sha256": source_sha,
                "native_runtime": runtime or {},
            }
            stream.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    total_items = valid = invalid = 0
    with output.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            total_items += 1
            for answer in row["answers"].values():
                if isinstance(answer, dict) and "invalid_reason" not in answer:
                    valid += 1
                else:
                    invalid += 1
    return {
        "model_id": model_id,
        "model_sha256": model_sha,
        "adapter_version": ADAPTER_VERSION,
        "adapter_sha256": adapter_sha,
        "source_file_sha256": source_sha,
        "output_sha256": file_digest(output),
        "input_items": len(prompts),
        "completed_items": total_items,
        "new_items": len(remaining),
        "valid_questions": valid,
        "invalid_questions": invalid,
        "complete": total_items == len(prompts),
        "native_runtime": runtime or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "parent-model",
        "source-path",
        "train-data",
        "train-manifest",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-id", default="decision2-kev-derived-4b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verified = verify_checkpoint(
        args.checkpoint,
        args.parent_model,
        args.source_path,
        args.train_data,
        args.train_manifest,
    )
    if args.verify_only:
        print(
            json.dumps(
                {
                    "model_sha256": verified["model_sha256"],
                    "rows": verified["rows"],
                    "parent_revision": verified["parent"]["model_revision"],
                },
                sort_keys=True,
            )
        )
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --verify-only")
    _, decide, runtime = load_native(args.checkpoint, args.source_path, args.device)
    result = collect(
        args.input,
        args.output,
        verified,
        model_id=args.model_id,
        decide=decide,
        device=args.device,
        resume=args.resume,
        max_items=args.max_items,
        runtime=runtime,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
