"""Gold-free native Laya continuation inference with full ancestry checks.

This adapter is research-only and never claims the continuation is a released
Decision 2.0 model. It applies the same truncation policy as the source Laya
baseline so the development scores remain comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from inference.laya import MODEL_REVISION, truncation_diagnostics, verify_release
from inference.run import digest, load_prompts, synchronize
from training.laya.build_research_data import sha


def verify_checkpoint(
    run: Path,
    checkpoint: Path,
    source_model: Path,
    source_code: Path,
    *,
    allow_partial: bool = False,
) -> dict:
    run = run.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    if checkpoint.parent != run:
        raise ValueError("Checkpoint must belong to the exact research run")
    complete_path, run_path = run / "COMPLETE.json", run / "RUN.json"
    complete, run_receipt = json.loads(complete_path.read_text()), json.loads(
        run_path.read_text()
    )
    if (
        complete.get("schema_version") != "decision2-laya-joint-research-complete/1"
        or (complete.get("status") != "complete" and not allow_partial)
        or complete.get("run_sha256") != sha(run_path)
        or run_receipt.get("schema_version") != "decision2-laya-joint-research-run/1"
        or run_receipt.get("research_only") is not True
    ):
        raise ValueError("Run is partial or its ancestry differs")
    base = verify_release(source_model, source_code, MODEL_REVISION)
    if run_receipt.get("base") != base:
        raise ValueError("Pinned base/source identity differs")
    point = next(
        (
            point
            for point in complete["curve"]
            if point.get("checkpoint")
            and checkpoint.name == f"checkpoint-{point['step']:06d}"
        ),
        None,
    )
    if point is None:
        raise ValueError("Checkpoint absent from completed selection curve")
    receipt_path = checkpoint / "CHECKPOINT.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema_version") != "decision2-laya-joint-checkpoint/1"
        or receipt.get("research_only") is not True
        or receipt.get("release_qualified") is not False
        or receipt.get("run_sha256") != sha(run_path)
        or receipt.get("step") != point["step"]
        or point["checkpoint"]["checkpoint_sha256"] != sha(receipt_path)
        or point["checkpoint"]["selection"] != receipt["selection"]
    ):
        raise ValueError("Research checkpoint receipt differs")
    for name, expected in receipt["files"].items():
        if sha(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint artifact differs: {name}")
    return {
        "run_sha256": sha(run_path),
        "complete_sha256": sha(complete_path),
        "checkpoint_sha256": sha(receipt_path),
        "model_weights_sha256": receipt["files"]["model.safetensors"],
        "model_config_sha256": receipt["files"]["rl_agent_config.json"],
        "step": receipt["step"],
        "base": base,
    }


def collect(
    *,
    run: Path,
    checkpoint: Path,
    source_model: Path,
    source_code: Path,
    prompts: Path,
    output: Path,
    max_items: int | None = None,
    allow_partial: bool = False,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    identity = verify_checkpoint(
        run, checkpoint, source_model, source_code, allow_partial=allow_partial
    )
    rows = load_prompts(prompts)
    if max_items is not None:
        rows = rows[:max_items]
    sys.path.insert(0, str(source_code))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from laya.agent import Agent
    from laya.common import encode_text, render_options, serialize_state

    if torch.version.hip is None or torch.cuda.device_count() != 1:
        raise RuntimeError("One visible ROCm GPU is required")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    agent = Agent(
        str(checkpoint),
        device="cuda:0",
        fast=False,
        compile=False,
        expected_sha256={
            "model.safetensors": identity["model_weights_sha256"],
            "rl_agent_config.json": identity["model_config_sha256"],
        },
    )
    if agent.device.type != "cuda":
        raise RuntimeError("Laya continuation silently fell back from ROCm")
    output.parent.mkdir(parents=True, exist_ok=True)
    valid, truncated, head_exceeded = 0, 0, 0
    with output.open("x") as stream:
        for row in rows:
            payload = {"state": row["state"], "questions": row["questions"]}
            diagnostics = truncation_diagnostics(
                agent,
                row["state"],
                row["questions"],
                encode_text,
                render_options,
                serialize_state,
            )
            synchronize("cuda:0")
            started = time.perf_counter()
            try:
                response = agent.system_one(**payload)
                error = None
            except ValueError as exc:
                if "options exceed head_max_len=" not in str(exc):
                    raise
                head_exceeded += 1
                error = "head_budget_exceeded"
                response = {
                    "answers": {
                        qid: {"type": q["type"], "error": error}
                        for qid, q in row["questions"].items()
                    },
                    "usage": None,
                    "model": "laya-joint-typed-research",
                }
            synchronize("cuda:0")
            answers = dict(response["answers"])
            if set(answers) != set(row["questions"]):
                raise ValueError("Native answer IDs differ")
            native_answers = {}
            if error is None:
                for qid, record in diagnostics["by_question"].items():
                    if any(
                        record[key]
                        for key in (
                            "instructions_truncated",
                            "options_truncated",
                            "state_truncated",
                        )
                    ):
                        native_answers[qid] = answers[qid]
                        answers[qid] = {
                            "type": row["questions"][qid]["type"],
                            "error": "native_input_truncated",
                        }
                if native_answers:
                    truncated += 1
                else:
                    valid += 1
            latency = (time.perf_counter() - started) * 1000
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Nonfinite native latency")
            result = {
                "id": row["id"],
                "answers": answers,
                "usage": response.get("usage"),
                "latency_ms": latency,
                "model": response.get("model"),
                "backend": "laya-joint-typed-research",
                "model_id": "llm-semantic-router/dev-2.0-0.5b-research",
                "model_revision": identity["checkpoint_sha256"],
                "model_config_sha256": identity["model_config_sha256"],
                "model_weights_sha256": identity["model_weights_sha256"],
                "source_revision": identity["base"]["source_revision"],
                "source_input_sha256": digest(payload),
                "native_truncation": diagnostics,
                "invalid_reason": error,
                "research_only": True,
                "release_qualified": False,
                **({"native_answers": native_answers} if native_answers else {}),
            }
            stream.write(
                json.dumps(
                    result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            stream.flush()
    receipt = {
        "schema_version": "decision2-laya-joint-inference/1",
        "research_only": True,
        "release_qualified": False,
        "run_identity": identity,
        "input_sha256": sha(prompts),
        "output_sha256": sha(output),
        "rows": len(rows),
        "valid": valid,
        "native_input_truncated": truncated,
        "head_budget_exceeded": head_exceeded,
        "scope": "gold-free development inference; no FINAL",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    receipt = collect(
        run=args.run,
        checkpoint=args.checkpoint,
        source_model=args.source_model,
        source_code=args.source_code,
        prompts=args.input,
        output=args.output,
        max_items=args.max_items,
        allow_partial=args.allow_partial,
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "rows",
                    "valid",
                    "native_input_truncated",
                    "head_budget_exceeded",
                    "output_sha256",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
