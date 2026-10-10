"""Decision 2.5 model for 🤗 Transformers (``trust_remote_code=True``).

``AutoModel.from_pretrained(repo, trust_remote_code=True)`` loads the repository through its own runtime
(``decision25_runtime.py``) and returns a model with ``system_one(state=..., questions={...}, images=[...])``
(0 to 4 images per request). The
repository is a standard ``Qwen3_5Model`` checkpoint plus a 255-way answer-code readout, so without
``trust_remote_code`` the same repository loads as the plain backbone. A directory without
``decision_config.json`` is not a Decision model and is loaded as a stock ``Qwen3_5Model``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

try:
    from .decision25_runtime import DEFAULT_BATCH_SIZE, Decision25
except ImportError:
    from decision25_runtime import DEFAULT_BATCH_SIZE, Decision25

HUB_OPTIONS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "token",
)
RUNTIME_OPTIONS = ("device", "batch_size", "verify")
# Options of Transformers' own weight loader that this model does not use.
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
DECISION_CONFIG = "decision_config.json"


def _device_name(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Not a device: {value!r}")
    if isinstance(value, int):
        return "cpu" if value < 0 else f"cuda:{value}"
    if isinstance(value, (str, torch.device)):
        return str(torch.device(value))
    raise ValueError(f"Not a device: {value!r}")


def _device(device: Any, device_map: Any) -> str | None:
    """One device from ``device`` / ``device_map``; None keeps the runtime default (cuda:0 if present)."""
    if isinstance(device_map, dict):
        if set(device_map) != {""}:
            raise ValueError(
                "Decision 2.5 models run on one device: pass a device name or {'': device}"
            )
        device_map = device_map[""]
    if device_map == "auto":
        device_map = None
    names = {_device_name(v) for v in (device, device_map) if v is not None}
    if len(names) > 1:
        raise ValueError("device and device_map name different devices")
    return names.pop() if names else None


def _commit(name_or_path: Any, config: Any, hub: dict[str, Any]) -> Any:
    """The commit the config came from, so that config, code and weights come from one revision."""
    revision = hub.get("revision")
    commit = getattr(revision, "resolved", None)
    if commit is None and getattr(config, "name_or_path", None) == str(name_or_path):
        commit = getattr(config, "_commit_hash", None)
    return commit or revision


def _is_decision(name_or_path: Any, revision: Any, hub: dict[str, Any]) -> bool:
    local = Path(os.fspath(name_or_path)).expanduser()
    if local.is_dir():
        return (local / DECISION_CONFIG).is_file()
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    options = {
        k: v
        for k, v in hub.items()
        if k != "revision" and v is not None and v is not False
    }
    try:
        hf_hub_download(
            str(name_or_path), DECISION_CONFIG, revision=revision, **options
        )
    except EntryNotFoundError:
        return False
    return True


class Decision25Model(PreTrainedModel):
    """A Decision 2.5 checkpoint behind System One: ``system_one(state=..., questions={...}, images=[...])``."""

    config_class = Qwen3_5Config
    base_model_prefix = "decision"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _no_split_modules = []

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.runtime: Decision25 | None = None
        self.post_init()

    def _init_weights(self, module: Any) -> None:
        """Every weight comes from the checkpoint; nothing is initialized here."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *model_args: Any,
        config: Qwen3_5Config | None = None,
        **kwargs: Any,
    ):
        """Load a Hub repository or a local download through the Decision 2.5 runtime.

        Hub options: ``revision``, ``cache_dir``, ``token``, ``local_files_only``, ``force_download``.
        ``device`` or ``device_map`` names one device (default: cuda:0 if a GPU is visible, else CPU).
        Runtime options: ``batch_size`` (questions per forward pass, default 8) and ``verify`` (``fast``,
        ``full`` or ``none``; checks the files against ``MODEL_MANIFEST.json``). Numerics are fixed by the
        checkpoint (BF16 backbone, FP32 readout), so ``dtype`` only takes None, "auto" or bfloat16.
        """
        original = dict(kwargs)
        hub = {k: kwargs.pop(k) for k in HUB_OPTIONS if k in kwargs}
        if kwargs.pop("subfolder", "") not in ("", None):
            raise ValueError("A Decision 2.5 checkpoint loads from the repository root")
        revision = _commit(pretrained_model_name_or_path, config, hub)
        if not _is_decision(pretrained_model_name_or_path, revision, hub):
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

            original.pop("trust_remote_code", None)
            return Qwen3_5Model.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **original
            )
        if model_args:
            raise TypeError("Decision 2.5 models take no positional model arguments")
        options = {k: kwargs.pop(k) for k in RUNTIME_OPTIONS if k in kwargs}
        device_map = kwargs.pop("device_map", None)
        for key in ("dtype", "torch_dtype"):
            if kwargs.pop(key, None) not in (None, "auto", "bfloat16", torch.bfloat16):
                raise ValueError(
                    f"{key}: Decision 2.5 numerics are fixed by the checkpoint (BF16 backbone, FP32 "
                    "readout); pass None or 'auto'"
                )
        if kwargs.pop("attn_implementation", None) not in (None, "sdpa"):
            raise ValueError("Decision 2.5 backbones use SDPA attention")
        loading_info = kwargs.pop("output_loading_info", False)
        for key in LOADER_FLAGS:
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(
                f"Unsupported keyword arguments for a Decision 2.5 model: {sorted(kwargs)}"
            )
        if config is None:
            config = Qwen3_5Config.from_pretrained(
                pretrained_model_name_or_path,
                **{k: v for k, v in hub.items() if v is not None},
            )
        runtime = Decision25.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            device=_device(options.get("device"), device_map),
            batch_size=options.get("batch_size", DEFAULT_BATCH_SIZE),
            verify=options.get("verify", "fast"),
            **{
                k: v
                for k, v in hub.items()
                if k in ("cache_dir", "token", "local_files_only", "force_download")
            },
        )
        model = cls(config)
        model.runtime = runtime
        model.backbone = runtime.backbone
        model.name_or_path = str(pretrained_model_name_or_path)
        model.eval()
        if loading_info:
            return model, {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        return model

    def _require(self) -> Decision25:
        if self.runtime is None:
            raise RuntimeError("Load the model with from_pretrained")
        return self.runtime

    @property
    def model_name(self) -> str:
        return self._require().model_name

    @property
    def max_input_tokens(self) -> int | None:
        return self._require().max_length

    @property
    def decision_config(self) -> dict[str, Any]:
        return self._require().config

    @property
    def manifest(self) -> dict[str, Any] | None:
        return self._require().manifest

    def system_one(
        self,
        *,
        state: Any,
        questions: dict[str, Any],
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Typed Choice / Noul / Score answers about one state: ``{"model", "answers", "usage"}``.

        ``questions`` maps question IDs to ``{"type": "choice" | "noul" | "score", "instructions": ...,
        "criteria": ...}``; a question over the input limit is answered ``max_length_exceeded``, never
        truncated. ``images``: up to 4 images every question sees (PIL images, local paths, http(s) URLs or
        base64 ``data:image/...`` URLs), placed before the text and read at up to 1.6 MP each.
        """
        return self._require().system_one(
            state=state, questions=questions, images=images
        )

    def forward(
        self,
        state: Any = None,
        questions: dict[str, Any] | None = None,
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        return self.system_one(state=state, questions=questions, images=images)

    def to(self, *args: Any, **kwargs: Any) -> Decision25Model:
        """Move to another device; numerics are fixed by the checkpoint, so dtype casts are refused."""
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None or memory_format is not None:
            raise TypeError(
                "Decision 2.5 numerics are fixed by the checkpoint; only the device can change"
            )
        if device is not None:
            self._require().to(str(device))
        return self

    def cuda(self, device: Any = None) -> Decision25Model:
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return self.to(device if device is not None else "cuda")

    def cpu(self) -> Decision25Model:
        return self.to("cpu")

    def _cast(self, *args: Any, **kwargs: Any) -> Decision25Model:
        raise TypeError(
            "Decision 2.5 numerics are fixed by the checkpoint; dtype casts are not supported"
        )

    half = float = bfloat16 = double = _cast

    def train(self, mode: bool = True) -> Decision25Model:
        if mode:
            raise RuntimeError(
                "Decision 2.5 models are inference-only here; train from the checkpoint files"
            )
        return super().train(False)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The repository itself is the package; copy it with "
            "huggingface_hub.snapshot_download(repo_id, local_dir=...)"
        )

    def push_to_hub(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Decision 2.5 packages are published by their release pipeline"
        )
