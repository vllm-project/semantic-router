"""Qwen-family Decision 2.0 checkpoints (full or base-bound LoRA) behind System One.

Model construction, prompt encoding, the candidate head and answer
normalization come from the vendored modules that produced the scored
predictions (``_vendor/dev2model``), byte for byte except where the manifest
records an import-only rewrite. A base-bound adapter's source files are pinned
by repository revision and SHA-256; a supplied or downloaded copy is verified
before use. GPU inference uses a BF16 backbone with an FP32 head; CPU uses FP32.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ._vendor.dev2model.data import file_sha256


def resolve_base(base: dict[str, Any], base_path: str | Path | None) -> Path:
    """Local directory holding exactly the pinned base files (downloaded if absent)."""
    expected = base["files_sha256"]
    if base_path is None:
        from huggingface_hub import snapshot_download

        base_path = snapshot_download(
            base["repo_id"], revision=base["revision"], allow_patterns=sorted(expected)
        )
    root = Path(base_path).resolve(strict=True)
    for name, digest in expected.items():
        if not (root / name).is_file() or file_sha256(root / name) != digest:
            raise ValueError(f"Base file differs from the pinned revision: {name}")
    return root


class QwenDecision:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Any,
        temperatures: dict,
        cap: int,
        torch: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperatures = temperatures
        self.cap = cap
        self.torch = torch

    @classmethod
    def load(
        cls,
        root: Path,
        manifest: dict[str, Any],
        *,
        device: str,
        base_path: str | Path | None,
        threads: int | None,
    ) -> QwenDecision:
        import torch

        from ._vendor.dev2model.calibration import load_calibration
        from ._vendor.dev2model.decision_model import DecisionModel
        from ._vendor.dev2model.infer import checkpoint_fingerprint

        if threads:
            torch.set_num_threads(threads)
        source = None
        if manifest["profile"] == "qwen-adapter":
            source = resolve_base(manifest["base"], base_path)
        metadata = json.loads(
            (root / "decision_config.json").read_text(encoding="utf-8")
        )
        residual = metadata.get("dec_residual") is not None
        if residual:
            from ._vendor.dev2model.dec_model import dec_fingerprint

            identity = dec_fingerprint(root, source)
        else:
            identity = checkpoint_fingerprint(root, source)
        if identity["model_sha256"] != manifest["identity"]["model_sha256"]:
            raise ValueError("Model identity differs from the scored checkpoint")
        calibration = manifest.get("calibration")
        if calibration:
            temperatures, report = load_calibration(
                root / calibration["file"], identity["model_sha256"]
            )
            if report.get("inference", {}).get("max_length") not in (
                None,
                manifest["max_input_tokens"],
            ):
                raise ValueError("Calibration context differs from the package limit")
        else:
            temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0}
        target = torch.device(device)
        if target.type == "cuda" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("A CUDA/ROCm BF16 GPU is required for GPU inference")
        if residual:
            from ._vendor.dev2model.dec_model import load_dec_checkpoint

            model, tokenizer = load_dec_checkpoint(root, source)
        else:
            model, tokenizer = DecisionModel.from_checkpoint(root, source_path=source)
        model = model.float().to(target).eval()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        return cls(
            model, tokenizer, target, temperatures, manifest["max_input_tokens"], torch
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def system_one(
        self, state: Any, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        from ._vendor.dev2model.data import canonical
        from ._vendor.dev2model.decision_model import collate, encode
        from ._vendor.dev2model.infer import (
            _api_json_payload,
            product_answer,
            question_to_row,
        )

        if not _api_json_payload(state):
            raise ValueError("state must be text, an object, or an array")
        try:
            canonical(state)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must contain JSON data") from exc
        item = {"id": "request", "state": state}
        answers: dict[str, dict[str, Any]] = {}
        jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        tokens = 0
        for qid, question in questions.items():
            try:
                row = question_to_row(item, qid, question)
                encoded = encode(row, self.tokenizer, self.cap)
            except ValueError as exc:
                reason = (
                    "max_length_exceeded"
                    if "exceeds max_length" in str(exc)
                    else "invalid_question"
                )
                answers[qid] = {
                    "type": (
                        question.get("type") if isinstance(question, dict) else None
                    ),
                    "error": reason,
                }
                continue
            tokens += len(encoded["ids"])
            jobs.append((qid, row, encoded))
        if not jobs:
            return answers, tokens
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        batch = {
            key: value.to(self.device) if self.torch.is_tensor(value) else value
            for key, value in collate(
                [encoded for _, _, encoded in jobs], pad_id
            ).items()
        }
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with self.torch.inference_mode(), autocast:
            logits = self.model(**batch)
        if len(logits) != len(jobs):
            raise RuntimeError("Model returned the wrong number of question answers")
        for (qid, row, encoded), values in zip(jobs, logits):
            try:
                answers[qid] = product_answer(
                    row["task_type"],
                    encoded["keys"],
                    values[: len(encoded["keys"])].float().cpu().tolist(),
                    self.temperatures[row["task_type"]],
                    [option["description"] for option in row["options"]],
                )
            except ValueError:
                answers[qid] = {
                    "type": row["task_type"],
                    "error": "invalid_model_output",
                }
        ordered = {qid: answers[qid] for qid in questions}
        return ordered, tokens
