"""Shared runtime and configuration helpers for multimodal-large training."""

import json
import os
import random
import sys
from importlib import metadata
from typing import Any

MIN_SENTENCE_TRANSFORMERS_MAJOR = 5


def _is_primary_process() -> bool:
    for env_name in ("ACCELERATE_PROCESS_INDEX", "RANK", "LOCAL_RANK"):
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return env_value in {"0", "-1"}
    return True


def configure_unbuffered_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def log_progress(message: str) -> None:
    if _is_primary_process():
        print(message, flush=True)


def load_yaml(path: str) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 - keep CLI help import-light

    with open(path, encoding="utf-8") as handle:
        rendered = os.path.expandvars(handle.read())
    if "${" in rendered:
        raise ValueError(f"Config contains unresolved environment variables: {path}")
    return yaml.safe_load(rendered)


def set_seed(seed: int) -> None:
    import torch  # noqa: PLC0415 - optional runtime dependency

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_sentence_transformers_version() -> str:
    try:
        version = metadata.version("sentence-transformers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install the project requirements in the trainer environment."
        ) from exc

    major = int(version.split(".", 1)[0])
    if major < MIN_SENTENCE_TRANSFORMERS_MAJOR:
        raise RuntimeError(
            f"sentence-transformers>={5}.0.0 is required for the native multimodal trainer path; found {version}."
        )
    return version


def normalize_mixed_precision(value: Any) -> str:
    if isinstance(value, bool):
        return "bf16" if value else "no"

    normalized = str(value or "bf16").strip().lower()
    if normalized in {"false", "off", "none", "no"}:
        return "no"
    if normalized in {"true", "on"}:
        return "bf16"
    if normalized not in {"no", "fp8", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision mode: {value}")
    return normalized


def write_status(output_dir: str, payload: dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "train_status.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)


def is_datacenter_tri_encoder_config(cfg: dict[str, Any]) -> bool:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    return bool(
        data_cfg.get("cache_dir")
        and model_cfg.get("text_encoder_name")
        and model_cfg.get("image_encoder_name")
        and model_cfg.get("audio_encoder_name")
    )
