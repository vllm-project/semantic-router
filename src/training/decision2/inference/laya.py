"""Native Laya typed-decisions collector with explicit truncation diagnostics."""

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

MODEL_ID = "convaiinnovations/laya-typed-decisions"
MODEL_REVISION = "1a793eb568e6718f15941d08f85432581df534e3"
SOURCE_REVISION = "4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"
ARTIFACT_SHA256 = {
    "rl_agent_config.json": "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8",
    "model.safetensors": "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
    "tokenizer/tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "tokenizer/tokenizer_config.json": "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1",
}
ADAPTER_VERSION = "laya-native-v2"


def verify_release(
    model_path: Path, source_path: Path, revision: str
) -> dict[str, Any]:
    if revision != MODEL_REVISION:
        raise ValueError(
            f"Laya typed-decisions requires exact HF revision {MODEL_REVISION}"
        )
    if not local_revision(model_path, revision):
        raise ValueError(
            "HF local-dir metadata is required to attest the exact revision"
        )
    for name, expected in ARTIFACT_SHA256.items():
        if file_digest(model_path / name) != expected:
            raise ValueError(f"Published Laya artifact hash mismatch: {name}")
    config = json.loads(
        (model_path / "rl_agent_config.json").read_text(encoding="utf-8")
    )
    if (
        config.get("model_name") != "laya-typed-decisions"
        or config.get("encoder") != "answerdotai/ModernBERT-large"
        or config.get("max_len") != 1024
        or config.get("head_max_len") != 256
    ):
        raise ValueError("Unexpected Laya typed-decisions configuration")
    commit = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != SOURCE_REVISION:
        raise ValueError("Laya source differs from the pinned reviewed revision")
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
        raise ValueError("Laya source has modified tracked files")
    if not (source_path / "laya/agent.py").is_file():
        raise FileNotFoundError("Laya native Agent API is missing")
    return {
        "backend": "laya",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "revision_attested": True,
        "source_revision": SOURCE_REVISION,
        "model_config_sha256": ARTIFACT_SHA256["rl_agent_config.json"],
        "model_weights_sha256": ARTIFACT_SHA256["model.safetensors"],
    }


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
    loaded = sys.modules.get("laya")
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if (
            loaded_file is None
            or Path(loaded_file).resolve().parent != (source_path / "laya").resolve()
        ):
            raise RuntimeError("A different Laya package is loaded in this process")
    sys.path.insert(0, str(source_path))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers
    from laya.agent import Agent
    from laya.common import encode_text, render_options, serialize_state

    if (
        torch.version.hip is None
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("Laya ROCm evaluation requires one visible AMD GPU")
    agent = Agent(
        str(model_path),
        device=device,
        fast=False,
        compile=False,
        expected_sha256=ARTIFACT_SHA256,
    )
    if agent.device.type != "cuda":
        raise RuntimeError("Laya silently fell back from ROCm to CPU during model load")
    return (
        agent,
        encode_text,
        render_options,
        serialize_state,
        {
            "torch": str(torch.__version__),
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "native_dtype": str(agent.dtype),
            "native_fast": False,
            "native_max_len": agent.cfg["max_len"],
            "native_head_max_len": agent.cfg["head_max_len"],
            "native_temperature": agent.temperature,
            "native_temperature_by_options": agent.temperature_by_options,
            "runtime_qualification": "unvalidated_rocm",
        },
    )


def truncation_diagnostics(
    agent,
    state: Any,
    questions: dict[str, Any],
    encode_text,
    render_options,
    serialize_state,
) -> dict[str, Any]:
    """Mirror the published packing budgets without altering its inference."""
    tok = agent.tok
    max_len = agent.cfg["max_len"]
    head_max_len = agent.cfg["head_max_len"]
    state_ids = encode_text(
        tok,
        serialize_state(state).replace(tok.mask_token, " "),
        add_special_tokens=False,
    )["input_ids"]
    per_question = {}
    for qid, qdef in questions.items():
        agent._check_question(qid, qdef)
        q = agent._to_internal(qdef)
        opts = render_options(q)
        instructions = str(q["ins"]).replace(tok.mask_token, " ")
        head_ids = encode_text(
            tok,
            f"{q['t']} question: {instructions}",
            add_special_tokens=False,
        )["input_ids"]
        full_option_lengths = [
            len(
                encode_text(
                    tok,
                    " " + option.replace(tok.mask_token, " "),
                    add_special_tokens=False,
                )["input_ids"]
            )
            for option in opts
        ]
        option_lengths = [1 + min(48, n) for n in full_option_lengths]
        budget = head_max_len - sum(option_lengths)
        if budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(option_lengths)))
            option_lengths = [min(n, per) for n in option_lengths]
            budget = head_max_len - sum(option_lengths)
        head_kept = min(len(head_ids), max(8, budget))
        fixed_tokens = 1 + head_kept + 1 + sum(option_lengths) + 1 + 1
        state_kept = min(len(state_ids), max(0, max_len - fixed_tokens))
        per_question[qid] = {
            "instructions_truncated": head_kept < len(head_ids),
            "options_truncated": any(
                kept - 1 < original
                for kept, original in zip(option_lengths, full_option_lengths)
            ),
            "state_tokens_original": len(state_ids),
            "state_tokens_kept": state_kept,
            "state_truncated": state_kept < len(state_ids),
        }
    return {
        "any_truncation": any(
            value["instructions_truncated"]
            or value["options_truncated"]
            or value["state_truncated"]
            for value in per_question.values()
        ),
        "by_question": per_question,
    }


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
    for row in rows:
        if any(
            not isinstance(q, dict) or q.get("type") not in {"choice", "noul", "score"}
            for q in row["questions"].values()
        ):
            raise ValueError(f"{row['id']}: Laya supports Choice, Noul and Score only")
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
    agent, encode_text, render_options, serialize_state, runtime = load_native(
        model_path,
        source_path,
        device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    truncated_count = 0
    head_budget_exceeded_count = 0
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            truncation = truncation_diagnostics(
                agent,
                row["state"],
                row["questions"],
                encode_text,
                render_options,
                serialize_state,
            )
            try:
                response = agent.system_one(**payload)
                invalid_reason = None
            except ValueError as exc:
                if "options exceed head_max_len=" not in str(exc):
                    raise
                invalid_reason = "head_budget_exceeded"
                head_budget_exceeded_count += 1
                response = {
                    "model": "laya-rl-agent",
                    "answers": {
                        qid: {"type": question["type"], "error": invalid_reason}
                        for qid, question in row["questions"].items()
                    },
                    "usage": None,
                }
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if agent.device.type != "cuda":
                raise RuntimeError(
                    "Laya silently fell back from ROCm to CPU during inference"
                )
            if truncation["any_truncation"]:
                truncated_count += 1
            if (
                not isinstance(response, dict)
                or not isinstance(response.get("answers"), dict)
                or response["answers"].keys() != row["questions"].keys()
                or not math.isfinite(latency_ms)
                or latency_ms < 0
            ):
                raise ValueError(f"{row['id']}: malformed native Laya response")
            truncated_ids = {
                qid
                for qid, diagnostic in truncation["by_question"].items()
                if any(
                    diagnostic[key]
                    for key in (
                        "instructions_truncated",
                        "options_truncated",
                        "state_truncated",
                    )
                )
            }
            answers = dict(response["answers"])
            native_answers = {}
            if invalid_reason is None:
                for qid in truncated_ids:
                    native_answers[qid] = answers[qid]
                    answers[qid] = {
                        "type": row["questions"][qid]["type"],
                        "error": "native_input_truncated",
                    }
            receipt = {
                **identity,
                "id": row["id"],
                "answers": answers,
                "usage": response.get("usage"),
                "latency_ms": latency_ms,
                "adapter_version": ADAPTER_VERSION,
                "source_input_sha256": digest(payload),
                "model": response.get("model"),
                "invalid_reason": invalid_reason,
                "native_truncation": truncation,
                **({"native_answers": native_answers} if native_answers else {}),
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
        "truncated_now": truncated_count,
        "head_budget_exceeded_now": head_budget_exceeded_count,
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
