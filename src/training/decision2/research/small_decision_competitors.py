"""Gold-free native collectors for two pinned 0.6B decision checkpoints.

This is a research adapter. Preflight is required before inference and treats
any input that the native loader would truncate as invalid. The model sees
only state and questions from the prompt file, never a target or panel label.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

A_REVISION = "7c28eee87fc7f7a44dacabca5e89ed9281c995c2"
B_REVISION = "b327ec5efb5fdbf8bfafa3b369720ac5f6434b05"
B_CODE_REVISION = "adb9a0457ef53d2dc0e106215315670f1f6da871"
A_CODE_HASHES = {
    "configuration_bouncy.py": "95bc23268a60f263f1fdbc12aefca5673e424329972d9c7c7784eb81bec061cf",
    "modeling_bouncy.py": "75ef00bc7fa3ef1099815ddcf05f28138f1b0974dc5b7c2f715c5c23213d3e62",
}
A_WEIGHT_HASH = "d14c8e55d74af9a37301f4e90601d637791f9ecd3973b6d514d1feeb0f74cb74"
A_TOKENIZER_HASH = "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
B_WEIGHT_HASH = "ad0b65098a40026a9c2b763125c45ec312fa4e11205c07b5eb32ad10d392e47e"
B_HEAD_HASH = "da1328e06c64789d334350975cb3350d8d2ef7c779f133cdd5b400feb835f233"
B_TOKENIZER_HASH = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def input_digest(row: dict) -> str:
    value = {"state": row["state"], "questions": row["questions"]}
    return hashlib.sha256(json_bytes(value).rstrip(b"\n")).hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt IDs")
    for row in rows:
        if set(row) != {"id", "state", "questions"}:
            raise ValueError("prompt has an unexpected field, possibly a target")
        if len(row["questions"]) != 1:
            raise ValueError("expected one typed question per row")
    return rows


def verify_model(arm: str, model_dir: Path, code_dir: Path | None) -> dict:
    expected = (
        {
            "model.safetensors": A_WEIGHT_HASH,
            "tokenizer.json": A_TOKENIZER_HASH,
            **A_CODE_HASHES,
        }
        if arm == "a"
        else {
            "model.safetensors": B_WEIGHT_HASH,
            "decision_head.safetensors": B_HEAD_HASH,
            "tokenizer.json": B_TOKENIZER_HASH,
        }
    )
    actual = {name: sha_file(model_dir / name) for name in expected}
    if actual != expected:
        raise ValueError("pinned model file hash mismatch")
    if arm == "b":
        if code_dir is None:
            raise ValueError("B requires its pinned source directory")
        revision = subprocess.check_output(
            ["git", "-C", str(code_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(code_dir), "status", "--porcelain"], text=True
        ).strip()
        if revision != B_CODE_REVISION or status:
            raise ValueError("B source revision or worktree differs from audit")
        actual["source_git_revision"] = revision
    return actual


def render_state(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False)
    raise ValueError("state must be a string, object or array")


def b_question(spec: dict) -> tuple[str, list[str], list[str]]:
    """Return native question, option strings, and original answer labels."""
    kind, instructions, criteria = spec["type"], spec["instructions"], spec["criteria"]
    if kind == "choice":
        labels = list(criteria)
        options = [f"{key}: {criteria[key]}" for key in labels]
    elif kind == "score":
        labels = [str(i) for i in range(len(criteria))]
        options = [f"{i}: {description}" for i, description in enumerate(criteria)]
    elif kind == "noul":
        labels, options = ["true", "false"], ["true", "false"]
        if criteria:
            instructions += (
                f"\nInterpret true as: {criteria['true']}"
                f"\nInterpret false as: {criteria['false']}"
            )
    else:
        raise ValueError(f"unsupported question type {kind}")
    if not 2 <= len(options) <= 26 or len(set(options)) != len(options):
        raise ValueError("native cardinality or option uniqueness")
    if any(
        not option.strip() or "\n" in option or "\r" in option for option in options
    ):
        raise ValueError("native option must be nonempty single-line text")
    return instructions, options, labels


def _enc(tok: Any, value: str) -> list[int]:
    return tok.encode(value, add_special_tokens=False)


def a_preflight(tok: Any, config: dict, row: dict) -> dict:
    spec = next(iter(row["questions"].values()))
    kind, criteria = spec["type"], spec["criteria"]
    if kind == "noul":
        options = ["no", "yes"]
        desc = (
            [criteria.get("false"), criteria.get("true")] if criteria else [None, None]
        )
    elif kind == "choice":
        options = list(criteria)
        desc = list(criteria.values())
    elif kind == "score":
        options = [str(i) for i in range(len(criteria))]
        desc = criteria
    else:
        raise ValueError("unsupported kind")
    block = _enc(tok, f"\n\nQuestion: {spec['instructions']}\nOptions:")
    for option, description in zip(options, desc):
        text = f"\n- {option}: {description}" if description else f"\n- {option}"
        block += _enc(tok, text)
    state = render_state(row["state"])
    bos = [int(tok.bos_token_id)] if tok.bos_token_id is not None else []
    state_ids = bos + _enc(tok, "State:\n" + state)
    budget = min(
        int(config["max_state_tokens"]), int(config["max_total_tokens"]) - len(block)
    )
    total = min(len(state_ids), max(0, budget)) + len(block)
    reason = None
    if budget < 16:
        reason = "native_budget"
    elif len(state_ids) > budget:
        reason = "native_state_truncation"
    elif total > 8192:
        reason = "native_dense_attention_cap"
    return {
        "state_tokens": len(state_ids),
        "question_tokens": len(block),
        "input_tokens": total,
        "reason": reason,
    }


def b_preflight(tok: Any, config: dict, row: dict) -> dict:
    spec = next(iter(row["questions"].values()))
    question, options, _ = b_question(spec)
    state = render_state(row["state"]).strip()
    state_ids = _enc(tok, state)
    prefix = _enc(tok, f"User: Context:\n{state}\n\n")
    suffix = (
        f"Question: {question.strip()}\nOptions:\n"
        + "\n".join(f"{chr(65+i)}) {option}" for i, option in enumerate(options))
        + "\nAnswer with the letter only.\nAssistant: The answer is"
    )
    suffix_ids = _enc(tok, suffix)
    total = len(prefix) + len(suffix_ids)
    reason = None
    if len(state_ids) > 1536:
        reason = "native_state_truncation"
    elif len(suffix_ids) > 448:
        reason = "native_question_truncation"
    elif total > int(config["max_position_embeddings"]):
        reason = "native_context_overflow"
    return {
        "state_tokens": len(state_ids),
        "question_tokens": len(suffix_ids),
        "input_tokens": total,
        "reason": reason,
    }


def preflight(
    arm: str,
    model_dir: Path,
    code_dir: Path | None,
    prompts: Path,
    output: Path,
    expected_prompts_sha: str,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    model_hashes = verify_model(arm, model_dir, code_dir)
    panel_hash = sha_file(prompts)
    if panel_hash != expected_prompts_sha:
        raise ValueError("prompt panel hash mismatch")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=(arm == "a")
    )
    config = json.loads((model_dir / "config.json").read_text())
    rows = read_rows(prompts)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter[str] = collections.Counter()
    with output.open("xb") as destination:
        for row in rows:
            result = (
                a_preflight(tok, config, row)
                if arm == "a"
                else b_preflight(tok, config, row)
            )
            result.update(id=row["id"], source_input_sha256=input_digest(row))
            counts[result["reason"] or "valid"] += 1
            destination.write(json_bytes(result))
    manifest = {
        "arm": arm,
        "revision": A_REVISION if arm == "a" else B_REVISION,
        "prompts_sha256": panel_hash,
        "preflight_sha256": sha_file(output),
        "model_hashes": model_hashes,
        "n": len(rows),
        "counts": dict(counts),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_bytes(
        json_bytes(manifest)
    )
    print(
        json.dumps(
            {
                "arm": arm,
                "n": len(rows),
                "counts": dict(counts),
                "preflight_sha256": manifest["preflight_sha256"],
            }
        )
    )


def _a_answer(model: Any, row: dict) -> tuple[dict, int]:
    response = model.score(state=row["state"], questions=row["questions"])
    return response["answers"], response["usage"]["input_tokens"]


def _b_answer(model: Any, row: dict) -> tuple[dict, int]:
    from rlcd.decide import ChoiceQ, NoulQ, ScoreQ

    question_id, spec = next(iter(row["questions"].items()))
    question, options, labels = b_question(spec)
    kind = spec["type"]
    native = (
        ChoiceQ(question, options)
        if kind == "choice"
        else ScoreQ(question, options) if kind == "score" else NoulQ(question)
    )
    state = render_state(row["state"])
    _, rendered = model.render(state, [native], 1536, 448)
    if rendered[0].context != state.strip() or rendered[0].question != question.strip():
        raise RuntimeError("native B loader truncated a preflight-valid row")
    answer = model.ask(state, [native])[0]
    if kind == "noul":
        normalized = {
            "type": "noul",
            "noul": answer["p_true"],
            "confidence": answer["confidence"],
        }
    else:
        probabilities = {
            label: answer["probs"][option] for label, option in zip(labels, options)
        }
        normalized = {
            "type": kind,
            "probabilities": probabilities,
            "confidence": answer["confidence"],
        }
        if kind == "choice":
            normalized["choice"] = labels[options.index(answer["value"])]
        else:
            normalized["score"] = answer["score"] * (len(labels) - 1)
    return {question_id: normalized}, -1


def run(
    arm: str,
    model_dir: Path,
    code_dir: Path | None,
    prompts: Path,
    preflight_path: Path,
    output: Path,
    expected_prompts_sha: str,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    model_hashes = verify_model(arm, model_dir, code_dir)
    panel_hash = sha_file(prompts)
    if panel_hash != expected_prompts_sha:
        raise ValueError("prompt panel hash mismatch")
    manifest = json.loads(
        preflight_path.with_suffix(preflight_path.suffix + ".manifest.json").read_text()
    )
    if (
        sha_file(preflight_path) != manifest["preflight_sha256"]
        or manifest["arm"] != arm
        or manifest["prompts_sha256"] != panel_hash
        or manifest["model_hashes"] != model_hashes
    ):
        raise ValueError("preflight identity mismatch")
    rows, planned = read_rows(prompts), [
        json.loads(x) for x in preflight_path.read_text().splitlines()
    ]
    if len(rows) != len(planned):
        raise ValueError("preflight length mismatch")
    for row, plan in zip(rows, planned):
        if row["id"] != plan["id"] or input_digest(row) != plan["source_input_sha256"]:
            raise ValueError("preflight row mismatch")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this research collector")
    if arm == "a":
        from transformers import AutoModel

        model = (
            AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True,
                dtype=torch.float32,
            )
            .to("cuda")
            .eval()
        )
    else:
        assert code_dir is not None
        sys.path.insert(0, str(code_dir))
        from rlcd.decide import Decider

        model = Decider.load(str(model_dir), device="cuda", fast=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    counts: collections.Counter[str] = collections.Counter()
    started = time.monotonic()
    with temporary.open("xb") as destination:
        for index, (row, plan) in enumerate(zip(rows, planned)):
            item_start = time.monotonic()
            reason = plan["reason"]
            if reason:
                answers, usage = {}, {
                    "input_tokens": plan["input_tokens"],
                    "output_tokens": 0,
                }
            else:
                try:
                    answers, actual_tokens = (
                        _a_answer(model, row) if arm == "a" else _b_answer(model, row)
                    )
                    if arm == "a" and actual_tokens != plan["input_tokens"]:
                        raise RuntimeError(
                            "native A token count disagrees with preflight"
                        )
                    usage = {
                        "input_tokens": (
                            plan["input_tokens"] if actual_tokens < 0 else actual_tokens
                        ),
                        "output_tokens": 0,
                    }
                except ValueError as error:
                    reason = f"native_value_error:{type(error).__name__}"
                    answers, usage = {}, {
                        "input_tokens": plan["input_tokens"],
                        "output_tokens": 0,
                    }
            if answers:
                answer = next(iter(answers.values()))
                if answer["type"] == "noul" and not 0 <= answer["noul"] <= 1:
                    reason, answers = "invalid_probability", {}
                elif answer["type"] in ("choice", "score") and (
                    not all(
                        math.isfinite(p) and 0 <= p <= 1
                        for p in answer["probabilities"].values()
                    )
                    or abs(sum(answer["probabilities"].values()) - 1) > 0.02
                ):
                    reason, answers = "invalid_distribution", {}
            counts[reason or "valid"] += 1
            result = {
                "id": row["id"],
                "source_input_sha256": plan["source_input_sha256"],
                "answers": answers,
                "latency_ms": (time.monotonic() - item_start) * 1000,
                "usage": usage,
            }
            if reason:
                result["invalid_reason"] = reason
            destination.write(json_bytes(result))
            if (index + 1) % 50 == 0:
                destination.flush()
                print(f"{arm}: {index + 1}/{len(rows)}", flush=True)
    os.replace(temporary, output)
    receipt = {
        "arm": arm,
        "revision": A_REVISION if arm == "a" else B_REVISION,
        "prompts_sha256": panel_hash,
        "preflight_sha256": manifest["preflight_sha256"],
        "predictions_sha256": sha_file(output),
        "model_hashes": model_hashes,
        "collector_sha256": sha_file(Path(__file__)),
        "n": len(rows),
        "counts": dict(counts),
        "duration_seconds": time.monotonic() - started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_bytes(
        json_bytes(receipt)
    )
    print(
        json.dumps(
            {
                "arm": arm,
                "n": len(rows),
                "counts": dict(counts),
                "predictions_sha256": receipt["predictions_sha256"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run"))
    parser.add_argument("--arm", choices=("a", "b"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--expected-prompts-sha", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight(
            args.arm,
            args.model_dir,
            args.code_dir,
            args.prompts,
            args.output,
            args.expected_prompts_sha,
        )
    else:
        if args.preflight is None:
            parser.error("--preflight is required for run")
        run(
            args.arm,
            args.model_dir,
            args.code_dir,
            args.prompts,
            args.preflight,
            args.output,
            args.expected_prompts_sha,
        )


if __name__ == "__main__":
    main()
