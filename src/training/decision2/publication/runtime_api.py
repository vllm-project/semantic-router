"""Portable typed-decision API copied into each Decision 2.0 model bundle.

This file becomes ``decision2/api.py`` in the published bundle. It imports
only sibling files copied with it and never refers to a training checkout.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "decision2-self-contained-bundle/1"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def verify_bundle(path: str | Path) -> dict[str, Any]:
    """Verify every packaged byte and the original merge/calibration lineage."""
    root = Path(path).resolve(strict=True)
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("bundle_version") != MANIFEST_VERSION
    ):
        raise ValueError("Unknown Decision 2.0 bundle manifest")
    files = manifest.get("files_sha256")
    model_files = manifest.get("model_files_sha256")
    if (
        not isinstance(files, dict)
        or not files
        or not isinstance(model_files, dict)
        or not model_files
    ):
        raise ValueError("Bundle lacks complete file hashes")
    for name, expected in files.items():
        if not isinstance(name, str):
            raise ValueError("Invalid bundle file manifest entry")
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or name == "MODEL_MANIFEST.json"
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError("Invalid bundle file manifest entry")
        file = root / relative
        if not file.is_file() or file.is_symlink() or _sha_file(file) != expected:
            raise ValueError(f"Bundle file hash mismatch: {name}")
    if any(
        files.get(f"model/{name}") != digest for name, digest in model_files.items()
    ):
        raise ValueError("Model files differ from the bundle file manifest")
    from .infer import checkpoint_fingerprint

    actual_model = checkpoint_fingerprint(root / "model")
    model_sha = actual_model["model_sha256"]
    if actual_model["files_sha256"] != model_files or model_sha != manifest.get(
        "model_sha256"
    ):
        raise ValueError("Bundle model hash differs from the materialized checkpoint")

    from .calibration import load_calibration, verified_materialization_origin

    origin = verified_materialization_origin(root / "model", model_sha)
    if origin is None or origin["source_model_sha256"] != manifest.get(
        "source_model_sha256"
    ):
        raise ValueError("Bundle is not the attested materialized LoRA checkpoint")
    temperatures, calibration = load_calibration(
        root / "calibration.json",
        model_sha,
        materialized_source_sha256=origin["source_model_sha256"],
    )
    if (
        temperatures != manifest.get("temperature_by_type")
        or _sha_file(root / "calibration.json") != manifest.get("calibration_sha256")
        or origin["receipt_sha256"] != manifest.get("materialization_sha256")
    ):
        raise ValueError("Bundle calibration or materialization identity mismatch")
    if calibration.get("inference", {}).get("max_length") != manifest.get("max_length"):
        raise ValueError("Bundle context limit differs from CAL inference contract")
    return manifest


class Decision2:
    """Native Choice, Noul and Score inference over a materialized checkpoint."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: Any,
        manifest: dict[str, Any],
        torch: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.manifest = manifest
        self.torch = torch
        self.max_length = manifest["max_length"]
        self.temperatures = manifest["temperature_by_type"]

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: str = "cuda:0") -> Decision2:
        root = Path(path).resolve(strict=True)
        manifest = verify_bundle(root)
        import torch

        from .decision_model import DecisionModel

        target = torch.device(device)
        if target.type == "cuda" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("A CUDA/ROCm BF16 GPU is required for GPU inference")
        model, tokenizer = DecisionModel.from_checkpoint(root / "model")
        model = model.float().to(target).eval()
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=target,
            manifest=manifest,
            torch=torch,
        )

    def system_one(
        self, *, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Answer independently supplied typed questions without truncation."""
        from .decision_model import collate, encode
        from .infer import normalized_answer, question_to_row

        if (
            not isinstance(questions, dict)
            or not questions
            or any(not isinstance(key, str) or not key for key in questions)
        ):
            raise ValueError("questions must be a nonempty mapping of question IDs")
        # This also rejects NaN and non-JSON state before the model sees it.
        _canonical(state)
        answers: dict[str, dict[str, Any]] = {}
        jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        usage_tokens = 0
        item = {"id": "request", "state": state}
        for qid, original in questions.items():
            question = dict(original) if isinstance(original, dict) else original
            if (
                isinstance(question, dict)
                and question.get("type") == "noul"
                and "criteria" not in question
            ):
                question["criteria"] = {"false": "No", "true": "Yes"}
            try:
                row = question_to_row(item, qid, question)
                encoded = encode(row, self.tokenizer, self.max_length)
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
            usage_tokens += len(encoded["ids"])
            jobs.append((qid, row, encoded))
        if jobs:
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
                raise RuntimeError(
                    "Model returned the wrong number of question answers"
                )
            for (qid, row, encoded), values in zip(jobs, logits):
                try:
                    answers[qid] = normalized_answer(
                        row["task_type"],
                        encoded["keys"],
                        values[: len(encoded["keys"])].float().cpu().tolist(),
                        self.temperatures[row["task_type"]],
                    )
                except ValueError:
                    answers[qid] = {
                        "type": row["task_type"],
                        "error": "invalid_model_output",
                    }
        return {
            "model": self.manifest["model_id"],
            "answers": answers,
            "usage": {"input_tokens": usage_tokens, "output_tokens": 0},
        }
