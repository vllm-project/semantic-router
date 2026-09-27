"""Self-contained native API for a directly trained full Qwen3 checkpoint.

This becomes ``decision2/api.py`` in the public package. The checkpoint is
copied byte-for-byte; no LoRA merge or materialization receipt is invented.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "decision2-official-qwen3-full/1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
MODEL_ID = "llm-semantic-router/DEV2.0-0.6B"
ARCHITECTURE = "qwen3-text-endpoints-global-query-shared-bilinear-mlp"
HUB_GITATTRIBUTES_SHA256 = (
    "4358a92acd019e7896d287637a9081061d85a0f5655f3308eaf79b0a0d5cbf91"
)


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


def _inventory(root: Path) -> dict[str, str]:
    """Check the exact public inventory, ignoring only import-created bytecode."""
    files = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        relative = Path(name)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        if name == ".gitattributes":
            if (
                not path.is_file()
                or path.is_symlink()
                or _sha_file(path) != HUB_GITATTRIBUTES_SHA256
            ):
                raise ValueError("Hub metadata differs from the pinned default")
            continue
        if path.is_dir():
            if path.is_symlink():
                raise ValueError("Package contains a directory link")
            continue
        if relative.parts[:2] == ("decision2", "__pycache__"):
            tag = re.escape(sys.implementation.cache_tag)
            match = re.fullmatch(
                rf"([A-Za-z_][A-Za-z_0-9]*)\.{tag}(?:\.opt-[012])?\.pyc", path.name
            )
            source = root / "decision2" / f"{match.group(1)}.py" if match else None
            if (
                len(relative.parts) == 3
                and path.is_file()
                and not path.is_symlink()
                and source is not None
                and source.is_file()
            ):
                continue
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError("Package contains an unsafe or nonregular file")
        files[name] = _sha_file(path)
    return files


def _tensor_count(path: Path) -> int:
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if size < 2 or size > 128 << 20:
            raise ValueError("Invalid safetensors header")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors tensors")
    tensors = [value for name, value in header.items() if name != "__metadata__"]
    if not tensors or any(
        not isinstance(value, dict)
        or not isinstance(value.get("shape"), list)
        or any(type(dim) is not int or dim < 1 for dim in value["shape"])
        for value in tensors
    ):
        raise ValueError("Invalid safetensors tensor shape")
    return sum(math.prod(value["shape"]) for value in tensors)


def verify_bundle(path: str | Path) -> dict[str, Any]:
    """Verify the complete package and selected full-checkpoint identity."""
    if Path(path).is_symlink():
        raise ValueError("Package root cannot be a symlink")
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
            or SHA.fullmatch(expected) is None
        ):
            raise ValueError("Invalid bundle file manifest entry")
    if _inventory(root) != {
        **files,
        "MODEL_MANIFEST.json": _sha_file(root / "MODEL_MANIFEST.json"),
    }:
        raise ValueError("Package inventory differs from its manifest")
    if any(
        files.get(f"model/{name}") != digest for name, digest in model_files.items()
    ):
        raise ValueError("Model files differ from the bundle file manifest")
    if {
        name.removeprefix("model/"): digest
        for name, digest in files.items()
        if name.startswith("model/")
    } != model_files:
        raise ValueError("Package has untracked model files")
    loader_files = manifest.get("loader_files_sha256")
    if (
        not isinstance(loader_files, dict)
        or {
            name.removeprefix("decision2/"): digest
            for name, digest in files.items()
            if name.startswith("decision2/")
        }
        != loader_files
    ):
        raise ValueError("Package loader source differs from manifest")
    from .infer import checkpoint_fingerprint

    actual_model = checkpoint_fingerprint(root / "model")
    model_sha = actual_model["model_sha256"]
    if actual_model["files_sha256"] != model_files or model_sha != manifest.get(
        "model_sha256"
    ):
        raise ValueError("Bundle model hash differs from the selected full checkpoint")

    metadata = json.loads(
        (root / "model/decision_config.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("architecture") != ARCHITECTURE
        or metadata.get("training_mode") != "full"
        or metadata.get("source_stage") != "base"
        or metadata.get("base_revision") != manifest.get("base_revision")
        or manifest.get("base_model") != "Qwen/Qwen3-0.6B-Base"
        or not isinstance(manifest.get("base_revision"), str)
        or REVISION.fullmatch(manifest["base_revision"]) is None
        or manifest.get("model_id") != MODEL_ID
        or type(metadata.get("text_parameter_count")) is not int
        or manifest.get("parameter_count")
        != metadata["text_parameter_count"]
        + _tensor_count(root / "model/decision_head.safetensors")
    ):
        raise ValueError("Package direct full-model provenance differs")
    from .calibration import load_calibration

    temperatures, calibration = load_calibration(root / "calibration.json", model_sha)
    if temperatures != manifest.get("temperature_by_type") or _sha_file(
        root / "calibration.json"
    ) != manifest.get("calibration_sha256"):
        raise ValueError("Bundle calibration identity mismatch")
    if calibration.get("inference", {}).get("max_length") != manifest.get("max_length"):
        raise ValueError("Bundle context limit differs from CAL inference contract")
    return manifest


class Decision2:
    """Native Choice, Noul and Score inference over an exact full checkpoint."""

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
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != manifest["parameter_count"]
        ):
            raise ValueError("Loaded parameter count differs from package manifest")
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
        from .infer import _api_json_payload, product_answer, question_to_row

        if (
            not isinstance(questions, dict)
            or not questions
            or any(not isinstance(key, str) or not key for key in questions)
        ):
            raise ValueError("questions must be a nonempty mapping of question IDs")
        if not _api_json_payload(state):
            raise ValueError("state must be text, an object, or an array")
        # Reject non-JSON and nonfinite state before the model sees it.
        try:
            _canonical(state)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must contain JSON data") from exc
        from .decision_model import collate, encode

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
        return {
            "model": self.manifest["model_id"],
            "answers": answers,
            "usage": {"input_tokens": usage_tokens, "output_tokens": 0},
        }
