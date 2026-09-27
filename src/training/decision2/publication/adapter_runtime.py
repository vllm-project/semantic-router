"""Portable, unmerged Decision 2.0 PEFT runtime.

Copied as ``decision2/api.py`` into an adapter package. Verification is CPU
only and deliberately does not fetch the external, immutable base model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

VERSION = "decision2-peft-adapter-package/1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
REQUIRED_PACKAGES = ("torch", "transformers", "peft", "safetensors", "huggingface_hub")
DTYPE_BYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(
    root: Path, *, ignore_cache: bool = False, ignore_bytecode: bool = False
) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Model directory is missing or is a symlink")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if ignore_cache and name.split("/")[0] == ".cache":
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Model directory contains a nonportable path")
        if path.is_dir():
            if path.is_symlink():
                raise ValueError("Model directory contains a directory symlink")
            continue
        if ignore_bytecode and relative.parts[:2] == ("decision2", "__pycache__"):
            tag = re.escape(sys.implementation.cache_tag)
            match = re.fullmatch(
                rf"([A-Za-z_][A-Za-z_0-9]*)\.{tag}(?:\.opt-[012])?\.pyc",
                path.name,
            )
            source = root / "decision2" / f"{match.group(1)}.py" if match else None
            if (
                len(relative.parts) == 3
                and not path.is_symlink()
                and path.is_file()
                and source is not None
                and source.is_file()
                and not source.is_symlink()
            ):
                continue
        # HF snapshots often link files to the content-addressed blob cache.
        # External files may be links, but the resolved bytes are always hashed.
        if not path.is_file() or (path.is_symlink() and not ignore_cache):
            raise ValueError("Model directory contains a nonregular package file")
        files[name] = _hash(path)
    return files


def _digest_map(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} is empty or malformed")
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or "\\" in name
            or not isinstance(digest, str)
            or SHA.fullmatch(digest) is None
        ):
            raise ValueError(f"{label} contains an unsafe entry")
    return value


def _dependencies(manifest: dict[str, Any]) -> None:
    lock = manifest.get("dependencies")
    if not isinstance(lock, dict) or set(lock) != {"python", *REQUIRED_PACKAGES}:
        raise ValueError("Package lacks a complete runtime dependency lock")
    if lock["python"] != ".".join(str(part) for part in sys.version_info[:3]):
        raise RuntimeError("Python version differs from the scored runtime")
    for package in REQUIRED_PACKAGES:
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Missing pinned runtime dependency: {package}") from exc
        if actual != lock[package]:
            raise RuntimeError(f"Runtime dependency differs from lock: {package}")


def _tensor_counts(path: Path) -> dict[str, int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        length = int.from_bytes(raw, "little")
        if not 2 <= length <= 128 << 20:
            raise ValueError("Invalid safetensors header length")
        data = stream.read(length)
        if len(data) != length:
            raise ValueError("Truncated safetensors header")
    header = json.loads(data)
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    payload_size = path.stat().st_size - 8 - length
    counts: dict[str, int] = {}
    intervals = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(tensor, dict):
            raise ValueError("Invalid safetensors tensor")
        shape, offsets, dtype = (
            tensor.get("shape"),
            tensor.get("data_offsets"),
            tensor.get("dtype"),
        )
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size < 0 for size in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or dtype not in DTYPE_BYTES
        ):
            raise ValueError("Invalid safetensors shape, offset or dtype")
        count = math.prod(shape)
        start, end = offsets
        if end - start != count * DTYPE_BYTES[dtype] or end > payload_size:
            raise ValueError("Safetensors payload differs from tensor header")
        counts[name] = count
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Safetensors payload has a gap or overlap")
        cursor = end
    if cursor != payload_size or not counts:
        raise ValueError("Safetensors payload is incomplete")
    return counts


def _parameter_breakdown(root: Path, source: Path) -> dict[str, int]:
    base = 0
    names: set[str] = set()
    weights = sorted(source.glob("*.safetensors"))
    if not weights or any(path.suffix == ".bin" for path in source.iterdir()):
        raise ValueError("Pinned base requires safetensors weights")
    for path in weights:
        for name, count in _tensor_counts(path).items():
            if name in names:
                raise ValueError("Duplicate base tensor in multiple shards")
            names.add(name)
            if name.startswith("model.language_model."):
                base += count
    adapter = sum(
        _tensor_counts(root / "model/adapter/adapter_model.safetensors").values()
    )
    head = sum(_tensor_counts(root / "model/decision_head.safetensors").values())
    if min(base, adapter, head) < 1:
        raise ValueError("Missing base, adapter or head parameters")
    return {
        "base_text": base,
        "adapter": adapter,
        "head": head,
        "total": base + adapter + head,
    }


def verify_bundle(
    path: str | Path, source_path: str | Path | None = None
) -> dict[str, Any]:
    """Verify package bytes and, when supplied, *all* external source bytes."""
    if Path(path).is_symlink():
        raise ValueError("Adapter package cannot be a symlink")
    root = Path(path).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Adapter package is not a regular directory")
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("bundle_version") != VERSION:
        raise ValueError("Unknown adapter package version")
    base = manifest.get("base")
    if (
        not isinstance(base, dict)
        or not isinstance(base.get("repo_id"), str)
        or HF_ID.fullmatch(base["repo_id"]) is None
        or not isinstance(base.get("revision"), str)
        or REVISION.fullmatch(base["revision"]) is None
    ):
        raise ValueError("Adapter package needs a pinned upstream repository commit")
    files = _digest_map(manifest.get("files_sha256"), "package files")
    if "MODEL_MANIFEST.json" in files or _inventory(root, ignore_bytecode=True) != {
        **files,
        "MODEL_MANIFEST.json": _hash(root / "MODEL_MANIFEST.json"),
    }:
        raise ValueError("Adapter package file inventory differs from manifest")
    model_files = _digest_map(manifest.get("model_files_sha256"), "model files")
    if {
        name.removeprefix("model/"): digest
        for name, digest in files.items()
        if name.startswith("model/")
    } != model_files:
        raise ValueError(
            "Packaged adapter/head/tokenizer files differ from model identity"
        )
    loader_files = _digest_map(manifest.get("loader_files_sha256"), "loader files")
    if {
        name.removeprefix("decision2/"): digest
        for name, digest in files.items()
        if name.startswith("decision2/")
    } != loader_files:
        raise ValueError("Packaged loader sources differ from manifest")
    expected_source = _digest_map(base.get("files_sha256"), "upstream source files")
    if source_path is not None:
        if Path(source_path).is_symlink():
            raise ValueError("Upstream source directory cannot be a symlink")
        source = Path(source_path).resolve(strict=True)
        if _inventory(source, ignore_cache=True) != expected_source:
            raise ValueError("Upstream source files differ from immutable adapter pin")
        from .infer import checkpoint_fingerprint

        identity = checkpoint_fingerprint(root / "model", source)
        if identity.get("model_sha256") != manifest.get("model_sha256"):
            raise ValueError("PEFT checkpoint and upstream base identity disagree")
        breakdown = _parameter_breakdown(root, source)
        metadata = json.loads(
            (root / "model/decision_config.json").read_text(encoding="utf-8")
        )
        if metadata.get("text_parameter_count") != breakdown["base_text"]:
            raise ValueError(
                "Pinned base parameter count differs from checkpoint metadata"
            )
        if breakdown != manifest.get("parameter_breakdown") or breakdown[
            "total"
        ] != manifest.get("parameter_count"):
            raise ValueError(
                "Full base/adapter/head parameter count differs from package"
            )
        from .calibration import load_calibration

        temperatures, report = load_calibration(
            root / "calibration.json", identity["model_sha256"]
        )
        if temperatures != manifest.get("temperature_by_type") or report.get(
            "inference", {}
        ).get("max_length") != manifest.get("max_length"):
            raise ValueError("Native calibration differs from package contract")
    if (
        not isinstance(manifest.get("model_sha256"), str)
        or SHA.fullmatch(manifest["model_sha256"]) is None
    ):
        raise ValueError("Adapter checkpoint fingerprint is missing")
    if (
        not isinstance(manifest.get("parameter_count"), int)
        or manifest["parameter_count"] < 1
    ):
        raise ValueError("Adapter package has no full-base parameter count")
    if files.get("calibration.json") != manifest.get("calibration_sha256"):
        raise ValueError("Calibration hash differs from package files")
    return manifest


class Decision2:
    """Native Choice/Noul/Score inference with the original, unmerged PEFT path."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Any,
        manifest: dict[str, Any],
        torch: Any,
    ):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.manifest, self.torch = manifest, torch

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        source_path: str | Path | None = None,
        device: str = "cuda:0",
    ) -> Decision2:
        root = Path(path).resolve(strict=True)
        manifest = verify_bundle(root)
        _dependencies(manifest)
        if source_path is None:
            from huggingface_hub import snapshot_download

            base = manifest["base"]
            source_path = snapshot_download(
                repo_id=base["repo_id"],
                revision=base["revision"],
                allow_patterns=sorted(base["files_sha256"]),
            )
        source = Path(source_path).resolve(strict=True)
        verify_bundle(root, source)
        import torch

        from .decision_model import DecisionModel

        target = torch.device(device)
        if target.type == "cuda" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("A CUDA/ROCm BF16 GPU is required for GPU inference")
        model, tokenizer = DecisionModel.from_checkpoint(
            root / "model", source_path=source
        )
        model = model.float().to(target).eval()
        return cls(model, tokenizer, target, manifest, torch)

    def system_one(
        self, *, state: Any, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        from .decision_model import collate, encode
        from .infer import normalized_answer, question_to_row

        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        answers: dict[str, Any] = {}
        jobs = []
        tokens = 0
        item = {"id": "request", "state": state}
        for qid, original in questions.items():
            if not isinstance(qid, str) or not qid:
                raise ValueError("question IDs must be nonempty strings")
            question = dict(original) if isinstance(original, dict) else original
            if (
                isinstance(question, dict)
                and question.get("type") == "noul"
                and "criteria" not in question
            ):
                question["criteria"] = {"false": "No", "true": "Yes"}
            try:
                row = question_to_row(item, qid, question)
                encoded = encode(row, self.tokenizer, self.manifest["max_length"])
            except ValueError as exc:
                answers[qid] = {
                    "type": (
                        question.get("type") if isinstance(question, dict) else None
                    ),
                    "error": (
                        "max_length_exceeded"
                        if "exceeds max_length" in str(exc)
                        else "invalid_question"
                    ),
                }
                continue
            tokens += len(encoded["ids"])
            jobs.append((qid, row, encoded))
        if jobs:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            if pad_id is None:
                raise ValueError("Tokenizer needs a pad or EOS token")
            batch = {
                key: value.to(self.device) if self.torch.is_tensor(value) else value
                for key, value in collate([job[2] for job in jobs], pad_id).items()
            }
            autocast = (
                self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with self.torch.inference_mode(), autocast:
                logits = self.model(**batch)
            if len(logits) != len(jobs):
                raise RuntimeError("Model returned a different number of answers")
            for (qid, row, encoded), values in zip(jobs, logits):
                try:
                    answers[qid] = normalized_answer(
                        row["task_type"],
                        encoded["keys"],
                        values[: len(encoded["keys"])].float().cpu().tolist(),
                        self.manifest["temperature_by_type"][row["task_type"]],
                    )
                except ValueError:
                    answers[qid] = {
                        "type": row["task_type"],
                        "error": "invalid_model_output",
                    }
        return {
            "model": self.manifest["model_id"],
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
