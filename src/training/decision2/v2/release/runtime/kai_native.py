"""Kai/Lex-family native export behind the System One API, on CPU or ROCm GPU.

Verification and loading follow the vendored Kai runtime (``decision_runtime``)
exactly: the pinned native manifest, every file hash, and the pinned runtime
source hashes are checked before any bundled code runs. Inference uses the
native FP32 ``predict`` with type-sorted physical batches of eight and the
native 8,192-token complete-input cap. Over-budget requests are answered with
explicit ``context_overflow`` errors; nothing is truncated. The only change
from the published loader is that a CPU device is accepted.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

from ._vendor.decision_inference._system_one import system_one_records
from ._vendor.decision_runtime.native import Native, verify_files

REQUEST_MODEL = "decision2-06b-native"
BATCH_SIZE = 8


class ContextOverflow(ValueError):
    """The complete native input exceeds the cap; the request is answered invalid."""


def _flags(torch: Any, threads: int | None) -> None:
    torch.set_num_threads(threads or 4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


def _load_native(directory: Path, manifest_sha256: str, device: str) -> Native:
    """``decision_runtime.load_native`` without its ROCm-only device assertion."""
    root, manifest = verify_files(directory, manifest_sha256)
    namespace = "_decision_native_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(
        namespace, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = package
    try:
        spec.loader.exec_module(package)
        artifacts = importlib.import_module(namespace + ".artifacts")
        contract = importlib.import_module(namespace + ".contract")
        api = importlib.import_module(namespace + ".model")
        declared = contract.validate_config(
            json.loads((root / "decision_config.json").read_text())
        )
        if (declared["arm"], declared["training_arm"]) != ("all22", "S22"):
            raise ValueError(
                "Only the complete three-path all22/S22 architecture is supported"
            )
        model, collator, cfg = artifacts.load_export(root, device=device)
        if (cfg["arm"], cfg["training_arm"]) != ("all22", "S22"):
            raise ValueError(
                "Only the complete three-path all22/S22 architecture is supported"
            )
        if type(model) is not api.DecisionModel or artifacts.verify_native(root) != (
            manifest,
            cfg,
        ):
            raise ValueError("Loaded class or native identity mismatch")
        return Native(
            root,
            manifest_sha256,
            model,
            collator,
            cfg,
            api,
            contract,
            artifacts,
            package_name=namespace,
        )
    except BaseException:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]
        raise


def _answer(
    row: dict[str, Any], prediction: dict[str, Any], cap: int
) -> dict[str, Any]:
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
        or not 1 <= int(prediction.get("input_tokens", 0)) <= cap
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


class KaiNative:
    def __init__(self, native: Native, device: Any, cap: int, torch: Any):
        self.native = native
        self.device = device
        self.cap = cap
        self.torch = torch

    @classmethod
    def load(
        cls, root: Path, manifest: dict[str, Any], *, device: str, threads: int | None
    ) -> KaiNative:
        import torch

        _flags(torch, threads)
        target = torch.device(device)
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA/ROCm device is visible")
            torch.cuda.set_device(target.index or 0)
        cap = manifest["max_input_tokens"]
        native = _load_native(
            root / "native", manifest["identity"]["native_manifest_sha256"], str(target)
        )
        if (
            native.collator.max_length != cap
            or native.collator.state_truncation != "error"
        ):
            raise ValueError(
                "Native collator must keep the complete-input cap without truncation"
            )
        return cls(native, target, cap, torch)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.native.model.parameters())

    def _rows(self, state: Any, questions: dict[str, Any]) -> list[dict[str, Any]]:
        rows = system_one_records(
            {"model": REQUEST_MODEL, "state": state, "questions": questions}
        )
        for index, row in enumerate(rows):
            row["id"] = f"systemone:0:{index}"
        return rows

    def _admit(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            try:
                item = self.native.collator.encode(row, labeled=False)
            except ValueError as exc:
                message = str(exc)
                if (
                    "no implicit truncation" in message
                    or "no room for state" in message
                ):
                    raise ContextOverflow(message) from exc
                raise
            if (
                item["state_tokens_original"] != item["state_tokens_kept"]
                or item["input_tokens"] > self.cap
            ):
                raise ContextOverflow("Native input exceeds the complete-input cap")

    def _predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._admit(rows)
        if self.native.training_state is not None:
            raise ValueError("Restore native policy before native inference")
        self.native.model.verify_inventory(check_values=False)
        order = sorted(range(len(rows)), key=lambda i: rows[i]["question"]["type"])
        predictions = self.native.model_api.predict(
            self.native.model,
            self.native.collator,
            [rows[i] for i in order],
            batch_size=BATCH_SIZE,
            device=str(self.device),
            precision="fp32",
        )
        restored: list[dict[str, Any] | None] = [None] * len(rows)
        for index, prediction in zip(order, predictions):
            restored[index] = prediction
        if any(item is None for item in restored):
            raise RuntimeError("Incomplete native prediction")
        return restored  # type: ignore[return-value]

    def system_one(
        self, state: Any, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        rows = self._rows(state, questions)
        try:
            predictions = self._predict(rows)
        except ContextOverflow:
            return (
                {
                    qid: {"type": question.get("type"), "error": "context_overflow"}
                    for qid, question in questions.items()
                },
                0,
            )
        answers = {
            row["question"]["id"]: _answer(row, p, self.cap)
            for row, p in zip(rows, predictions)
        }
        return answers, sum(p["input_tokens"] for p in predictions)
