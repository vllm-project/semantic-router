# Copyright 2026 The vLLM Semantic Router Authors.
# SPDX-License-Identifier: Apache-2.0
"""Decision 1.0 model for 🤗 Transformers (``trust_remote_code=True``).

``AutoModel.from_pretrained(repo, trust_remote_code=True)`` downloads the files
named by the repository's ``config.json`` and builds the Decision 1.0 native
inference path from the modules next to this one. ``system_one(state=...,
questions={...})`` answers typed Choice, Noul and Score questions with the System
One response body; there is no text generation and no chat API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel

from .configuration_decision1 import Decision1Config
from .decision1_system_one import (
    DecisionInputError,
    DecisionInputTooLongError,
    answer,
    build_row,
    error_answer,
    validate_question,
    validate_request,
)

HUB_OPTIONS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "token",
)
RUNTIME_OPTIONS = ("device", "threads")
# Options of Transformers' own weight loader, which this model does not use.
LOADER_FLAGS = (
    "trust_remote_code",
    "_from_auto",
    "_from_pipeline",
    "adapter_kwargs",
    "code_revision",
    "_commit_hash",
    "low_cpu_mem_usage",
    "use_safetensors",
    "resume_download",
    "user_agent",
)
# Input limits and prompt policies of the published models, as the Decision runtime serves them.
FAMILY_PROFILES = {
    "vela-encoder": {"max_input_tokens": 1024, "choice_null_description": "render_key"},
    "qwen3.5-decision": {
        "max_input_tokens": 16384,
        "choice_null_description": "preserve_json_null",
    },
}
MODEL_PROFILES = {"Decision-1.0-Nox-4B": {"choice_null_description": "render_key"}}

__all__ = ["Decision1Model", "DecisionInputError", "DecisionInputTooLongError"]


def _device_name(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Not a device: {value!r}")
    if isinstance(value, int):
        return "cpu" if value < 0 else f"cuda:{value}"
    if isinstance(value, (str, torch.device)):
        return str(torch.device(value))
    raise ValueError(f"Not a device: {value!r}")


def _device(device: Any, device_map: Any) -> torch.device:
    if isinstance(device_map, dict):
        if set(device_map) != {""}:
            raise ValueError(
                "Decision 1.0 models run on one device: pass a device name or {'': device}"
            )
        device_map = device_map[""]
    if device_map == "auto":
        device_map = None
    names = {_device_name(v) for v in (device, device_map) if v is not None}
    if len(names) > 1:
        raise ValueError("device and device_map name different devices")
    if names:
        return torch.device(names.pop())
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _offline(options: dict[str, Any]) -> dict[str, Any]:
    """With HF_HUB_OFFLINE, read the cache only (huggingface_hub would still list a commit's files)."""
    try:
        from huggingface_hub import is_offline_mode

        offline = is_offline_mode()
    except ImportError:
        from huggingface_hub import constants

        offline = constants.HF_HUB_OFFLINE
    return {**options, "local_files_only": True} if offline else options


def _package_dir(
    name_or_path: Any, config: Decision1Config, hub: dict[str, Any]
) -> Path:
    """The repository revision as a directory: a local download, or a snapshot in the Hugging Face cache."""
    local = Path(os.fspath(name_or_path)).expanduser()
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    revision = hub.get("revision")
    # Weights come from the commit the config came from: Transformers 5.18 passes a
    # ResolvedRevision, earlier versions record the commit on the config.
    commit = getattr(revision, "resolved", None)
    if commit is None and getattr(config, "name_or_path", None) == str(name_or_path):
        commit = getattr(config, "_commit_hash", None)
    options = _offline(
        {
            k: v
            for k, v in hub.items()
            if k != "revision" and v is not None and v is not False
        }
    )
    return Path(
        snapshot_download(
            str(name_or_path),
            revision=commit or revision,
            allow_patterns=config.files(),
            **options,
        )
    )


class Decision1Model(PreTrainedModel):
    """A Decision 1.0 model behind System One: ``system_one(state=..., questions={...})``."""

    config_class = Decision1Config
    base_model_prefix = "decision"
    main_input_name = "state"
    supports_gradient_checkpointing = False
    _no_split_modules: list[str] = []

    def __init__(self, config: Decision1Config):
        super().__init__(config)
        self.runtime = None
        self._source: Path | None = None
        self._threads: int | None = None
        self.post_init()

    def _init_weights(self, module: Any) -> None:
        """Every weight comes from the repository files; nothing is initialized here."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *model_args: Any,
        config: Decision1Config | None = None,
        **kwargs: Any,
    ) -> Decision1Model:
        """Load a Hub repository or a local download on one device.

        Hub options: ``revision``, ``cache_dir``, ``token``, ``local_files_only``,
        ``force_download``. ``device`` or ``device_map`` names one device (default:
        cuda:0 if a GPU is visible, else CPU); ``threads`` sets CPU threads.
        Numerics are the model's own, so ``dtype`` is only None or "auto".
        """
        if model_args:
            raise TypeError("Decision 1.0 models take no positional model arguments")
        hub = {k: kwargs.pop(k) for k in HUB_OPTIONS if k in kwargs}
        if kwargs.pop("subfolder", "") not in ("", None):
            raise ValueError("A Decision 1.0 repository loads from its root")
        options = {k: kwargs.pop(k) for k in RUNTIME_OPTIONS if k in kwargs}
        device_map = kwargs.pop("device_map", None)
        for key in ("dtype", "torch_dtype"):
            if kwargs.pop(key, None) not in (None, "auto"):
                raise ValueError(
                    f"{key}: Decision 1.0 numerics are fixed (FP32 encoders; decoders BF16 on a "
                    "GPU with an FP32 head, FP32 on CPU); pass None or 'auto'"
                )
        if kwargs.pop("attn_implementation", None) not in (None, "sdpa"):
            raise ValueError("Decision 1.0 models use SDPA attention")
        loading_info = kwargs.pop("output_loading_info", False)
        for key in LOADER_FLAGS:
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(
                f"Unsupported keyword arguments for a Decision 1.0 model: {sorted(kwargs)}"
            )
        if config is None:
            config = Decision1Config.from_pretrained(
                pretrained_model_name_or_path,
                **{k: v for k, v in hub.items() if v is not None},
            )
        model = cls(config)
        model._source = _package_dir(pretrained_model_name_or_path, config, hub)
        model._threads = options.get("threads")
        model._load(_device(options.get("device"), device_map))
        model.name_or_path = str(pretrained_model_name_or_path)
        if loading_info:
            return model, {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        return model

    def _load(self, device: torch.device) -> None:
        if self._threads:
            torch.set_num_threads(self._threads)
        descriptor = self.config.descriptor()
        profile = dict(FAMILY_PROFILES[descriptor["runtime_family"]])
        profile.update(MODEL_PROFILES.get(descriptor["model_name"], {}))
        if descriptor["runtime_family"] == "vela-encoder":
            from .decision1_vela import VelaRuntime

            runtime = VelaRuntime.load(
                self._source,
                descriptor,
                max_input_tokens=profile["max_input_tokens"],
                device=device,
            )
        else:
            from .decision1_qwen import QwenRuntime

            runtime = QwenRuntime.load(
                self._source,
                descriptor,
                max_input_tokens=profile["max_input_tokens"],
                choice_null_description=profile["choice_null_description"],
                device=device,
            )
        self._modules.pop("decision", None)
        self.decision = runtime.model
        self.runtime = runtime
        super().train(False)

    def _require(self) -> Any:
        if self.runtime is None:
            raise RuntimeError("Load the model with from_pretrained")
        return self.runtime

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def max_input_tokens(self) -> int:
        return self._require().max_input_tokens

    def system_one(self, *, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Typed Choice / Noul / Score answers about one state: ``{"model", "answers", "usage"}``.

        ``questions`` maps question IDs to ``{"type": "choice" | "noul" | "score",
        "instructions": ..., "criteria": ...}``. A malformed question is answered with
        ``invalid_question``. As in the native runtime, a request is admitted only when
        every question fits the input limit; otherwise each question is answered with
        ``max_length_exceeded`` and nothing is truncated.
        """
        runtime = self._require()
        state = validate_request(state, questions)
        answers: dict[str, Any] = {}
        rows = []
        for question_id, question in questions.items():
            try:
                checked = validate_question(question_id, question)
            except DecisionInputError:
                answers[question_id] = error_answer(question, "invalid_question")
                continue
            rows.append(
                build_row(
                    question_id,
                    state,
                    checked,
                    noul_default_false=runtime.noul_default_false,
                    noul_default_true=runtime.noul_default_true,
                    noul_explicit_null=runtime.noul_explicit_null,
                )
            )
        tokens = 0
        if rows:
            try:
                probabilities, counts = runtime.predict(rows)
            except DecisionInputTooLongError:
                for row in rows:
                    answers[row.question_id] = {
                        "type": row.type,
                        "error": "max_length_exceeded",
                    }
            else:
                tokens = sum(counts)
                for row, values in zip(rows, probabilities):
                    answers[row.question_id] = answer(row, values)
        return {
            "model": self.config.model_name,
            "answers": {question_id: answers[question_id] for question_id in questions},
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    def forward(
        self, state: Any = None, questions: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.system_one(state=state, questions=questions)

    def to(self, *args: Any, **kwargs: Any) -> Decision1Model:
        """Move to another device by loading the repository there through the same path."""
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None or memory_format is not None:
            raise TypeError(
                "Decision 1.0 numerics are fixed; only the device can change"
            )
        if device is None:
            return self
        current = next(self.decision.parameters()).device
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device != current:
            self._load(device)
        return self

    def cuda(self, device: Any = None) -> Decision1Model:
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return self.to(device if device is not None else "cuda")

    def cpu(self) -> Decision1Model:
        return self.to("cpu")

    def _cast(self, *args: Any, **kwargs: Any) -> Decision1Model:
        raise TypeError(
            "Decision 1.0 numerics are fixed; dtype casts are not supported"
        )

    def half(self, *args: Any, **kwargs: Any) -> Decision1Model:
        return self._cast()

    def float(self, *args: Any, **kwargs: Any) -> Decision1Model:
        return self._cast()

    def bfloat16(self, *args: Any, **kwargs: Any) -> Decision1Model:
        return self._cast()

    def double(self, *args: Any, **kwargs: Any) -> Decision1Model:
        return self._cast()

    def train(self, mode: bool = True) -> Decision1Model:
        if mode:
            raise RuntimeError("Decision 1.0 models are inference-only")
        return super().train(False)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The repository itself is the model; copy it with "
            "huggingface_hub.snapshot_download(repo_id, local_dir=...)"
        )

    def push_to_hub(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Decision 1.0 repositories are published by their release"
        )
