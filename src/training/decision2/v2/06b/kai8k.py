"""Kai/Lex native bundles at their unchanged 8,192-token native cap.

The published SystemOne wrapper admits at most 1,024 complete tokens; the
native collator and training API accept 8,192. This module keeps the bundle's
own conversion, packing, FP32 inference and exporter, and only removes the
product 1K guard. Inputs above 8,192 tokens still fail; nothing is truncated.
For inputs within 1,024 tokens the outputs equal the published profile.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from inference.kai_lex import MODELS, verify_native_bundle

from .common import MAX_INPUT_TOKENS, import_bundle

REQUEST_MODEL = "decision2-06b-native"


class ContextOverflow(ValueError):
    """The complete native input exceeds the cap; the answer is invalid."""


def runtime_flags(torch: Any) -> None:
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


def load(
    bundle: str | Path,
    backend: str,
    *,
    native_dir: str | Path | None = None,
    manifest_sha256: str | None = None,
    device: str = "cuda:0",
) -> tuple[Any, dict[str, Any]]:
    """Verify the pinned bundle, then load its native or a compatible export."""
    bundle = Path(bundle).resolve(strict=True)
    identity = verify_native_bundle(bundle, backend, MODELS[backend]["revision"])
    import_bundle(bundle)
    import torch
    from decision_runtime import load_native

    runtime_flags(torch)
    torch.cuda.set_device(0)
    target = Path(native_dir) if native_dir is not None else bundle / "native"
    expected = manifest_sha256 or MODELS[backend]["manifest_sha256"]
    native = load_native(target, expected_manifest_sha256=expected, device=device)
    if (
        native.collator.max_length != MAX_INPUT_TOKENS
        or native.collator.state_truncation != "error"
    ):
        raise ValueError(
            "Native collator must keep the 8,192 complete-input cap without truncation"
        )
    identity = {
        **identity,
        "native_dir": str(target),
        "native_manifest_sha256": expected,
    }
    return native, identity


def request_rows(state: Any, questions: dict[str, Any]) -> list[dict[str, Any]]:
    """One System One request -> native rows, exactly as SystemOne.batch orders them."""
    from decision_inference._system_one import system_one_records

    rows = system_one_records(
        {"model": REQUEST_MODEL, "state": state, "questions": questions}
    )
    for index, row in enumerate(rows):
        row["id"] = f"systemone:0:{index}"
    return rows


def admit(native: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Encode every row before any forward; overlength input is an explicit failure."""
    encoded = []
    for row in rows:
        try:
            item = native.collator.encode(row, labeled=False)
        except ValueError as exc:
            message = str(exc)
            if "no implicit truncation" in message or "no room for state" in message:
                raise ContextOverflow(message) from exc
            raise
        if (
            item["state_tokens_original"] != item["state_tokens_kept"]
            or item["input_tokens"] > MAX_INPUT_TOKENS
        ):
            raise ContextOverflow("Native input exceeds the complete-input cap")
        encoded.append(item)
    return encoded


def predict_rows(native: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Type-sorted physical batches of eight, as the published SystemOne default."""
    from decision_runtime import predict

    admit(native, rows)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["question"]["type"])
    predictions = predict(native, [rows[i] for i in order], batch_size=8)
    restored: list[dict[str, Any] | None] = [None] * len(rows)
    for index, prediction in zip(order, predictions):
        restored[index] = prediction
    if any(item is None for item in restored):
        raise RuntimeError("Incomplete native prediction")
    return restored  # type: ignore[return-value]


def answer(row: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Published `_answer` semantics with the 8,192 native cap instead of 1,024."""
    question = row["question"]
    kind = question["type"].lower()
    ids = (
        ["no", "yes"]
        if kind == "noul"
        else [v["id"] for v in question["options" if kind == "choice" else "levels"]]
    )
    if (
        prediction.get("id") != row["id"]
        or prediction.get("question_id") != question["id"]
        or prediction.get("candidate_ids") != ids
        or not 1 <= int(prediction.get("input_tokens", 0)) <= MAX_INPUT_TOKENS
        or prediction.get("state_tokens_original")
        != prediction.get("state_tokens_kept")
    ):
        raise RuntimeError("Prediction identity or complete-input mismatch")
    p = prediction["probabilities"]
    if (
        len(p) != len(ids)
        or not all(math.isfinite(v) and 0 <= v <= 1 for v in p)
        or abs(sum(p) - 1) > 2e-5
    ):
        raise RuntimeError("Invalid prediction probabilities")
    if kind == "noul":
        return {"type": "noul", "noul": p[1]}
    best = ids[max(range(len(p)), key=p.__getitem__)]
    result: dict[str, Any] = {
        "type": kind,
        "probabilities": dict(zip(ids, p)),
        "confidence": max(p),
    }
    if kind == "choice":
        result["choice"] = best
    else:
        result["score"] = prediction["score"]
        result["legend"] = {v["id"]: v["text"] for v in question["levels"]}
    return result


def system_one(native: Any, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    rows = request_rows(state, questions)
    predictions = predict_rows(native, rows)
    return {
        "answers": {
            row["question"]["id"]: answer(row, p) for row, p in zip(rows, predictions)
        },
        "input_tokens": sum(p["input_tokens"] for p in predictions),
    }


def record_probabilities(
    native: Any, records: list[dict[str, Any]]
) -> list[list[float]]:
    """Labeled native records (targets ignored) -> native-order probabilities."""
    rows = [
        {k: v for k, v in record.items() if k not in ("target", "hard_target_id")}
        for record in records
    ]
    for index, row in enumerate(rows):
        row["id"] = f"record:{index}"
    return [p["probabilities"] for p in predict_rows(native, rows)]
