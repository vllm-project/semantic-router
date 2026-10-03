"""Decision 2.0 model for 🤗 Transformers (``trust_remote_code=True``).

``AutoModel.from_pretrained(repo, trust_remote_code=True)`` loads the repository through its own System One
runtime, the ``decision2/`` package that ``decision2.Decision2.from_pretrained`` runs natively: every file is
checked against ``MODEL_MANIFEST.json``, the loaded parameter count and the scored model identity are asserted,
and a base-bound adapter's pinned base is fetched at its pinned revision and checked file by file. Answers,
numerics and errors are the native runtime's; there is no text generation.

Transformers copies only flat ``*.py`` files next to ``config.json`` into its dynamic-module cache, so the
runtime package is copied from the downloaded revision into the same directory (each file checked against the
manifest) and imported relative to this module. The runtime refuses links, while the Hugging Face cache stores
files as links, so a downloaded revision is loaded from a temporary hard-link view of it.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel

from .configuration_decision2 import Decision2Config

HUB_OPTIONS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "token",
)
RUNTIME_OPTIONS = (
    "device",
    "base_path",
    "threads",
    "bf16_resident",
    "graphs",
    "kernels",
    "share_context",
)
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
QWEN3_5_MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
# Gated-delta functions that Transformers binds at import to these packages' GPU-only kernels.
GATED_DELTA = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)
KERNEL_PACKAGES = ("fla", "causal_conv1d")
RUNTIME_DIR = "decision2/"
_LOCK = threading.RLock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _device_name(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Not a device: {value!r}")
    if isinstance(value, int):
        return "cpu" if value < 0 else f"cuda:{value}"
    if isinstance(value, (str, torch.device)):
        return str(torch.device(value))
    raise ValueError(f"Not a device: {value!r}")


def _device(device: Any, device_map: Any) -> str | None:
    """One device from ``device`` / ``device_map``; None keeps the runtime default."""
    if isinstance(device_map, dict):
        if set(device_map) != {""}:
            raise ValueError(
                "Decision 2.0 models run on one device: pass a device name or {'': device}"
            )
        device_map = device_map[""]
    if device_map == "auto":
        device_map = None
    names = {_device_name(v) for v in (device, device_map) if v is not None}
    if len(names) > 1:
        raise ValueError("device and device_map name different devices")
    return names.pop() if names else None


def _same_device(left: torch.device, right: torch.device) -> bool:
    def canonical(device: torch.device) -> torch.device:
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    return canonical(left) == canonical(right)


def _supported(function: Any, options: dict[str, Any], what: str) -> dict[str, Any]:
    accepted = inspect.signature(function).parameters
    unknown = sorted(k for k in options if k not in accepted)
    if unknown:
        raise TypeError(f"{what} does not take {unknown}")
    return options


def _offline(options: dict[str, Any]) -> dict[str, Any]:
    """With HF_HUB_OFFLINE, read the cache only (huggingface_hub would still list a commit's files)."""
    try:
        from huggingface_hub import is_offline_mode

        offline = is_offline_mode()
    except ImportError:
        from huggingface_hub import constants

        offline = constants.HF_HUB_OFFLINE
    return {**options, "local_files_only": True} if offline else options


def _package_dir(name_or_path: Any, config: Any, hub: dict[str, Any]) -> Path:
    """The repository revision as a directory: a local download, or a snapshot in the Hugging Face cache."""
    local = Path(os.fspath(name_or_path)).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    revision = hub.get("revision")
    # The weights come from the commit the config came from: Transformers 5.18 passes a
    # ResolvedRevision, Transformers 5.17 records the commit on the config.
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
    _supported(snapshot_download, options, "huggingface_hub.snapshot_download")
    return Path(
        snapshot_download(str(name_or_path), revision=commit or revision, **options)
    )


def _pinned_base(base: dict[str, Any], hub: dict[str, Any]) -> str:
    """The adapter's pinned base files at the pinned revision, in the same cache (checked by the runtime)."""
    from huggingface_hub import snapshot_download

    options = _offline(
        {
            k: hub[k]
            for k in ("cache_dir", "local_files_only", "token")
            if hub.get(k) is not None
        }
    )
    return snapshot_download(
        base["repo_id"],
        revision=base["revision"],
        allow_patterns=sorted(base["files_sha256"]),
        **options,
    )


def _link_free(root: Path) -> tuple[Path, Path | None]:
    """``root`` if it holds regular files only, else a temporary hard-link view of it (to be removed)."""
    files = [p for p in sorted(root.rglob("*")) if not p.is_dir()]
    if not root.is_symlink() and not any(p.is_symlink() for p in files):
        return root, None
    # Next to the blobs of a cache snapshot (same file system); elsewhere next to the directory.
    parent = root.parent.parent if root.parent.name == "snapshots" else root.parent
    try:
        view = Path(tempfile.mkdtemp(prefix=".decision2-view-", dir=parent))
    except OSError:
        view = Path(tempfile.mkdtemp(prefix="decision2-view-"))
    try:
        for source in files:
            target = view / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            real = source.resolve(strict=True)
            try:
                os.link(real, target)
            except OSError:
                shutil.copyfile(real, target)
    except BaseException:
        shutil.rmtree(view, ignore_errors=True)
        raise
    return view, view


def _runtime_matches(target: Path, files: dict[str, str]) -> bool:
    return all(
        (target / name[len(RUNTIME_DIR) :]).is_file()
        and _sha256(target / name[len(RUNTIME_DIR) :]) == digest
        for name, digest in files.items()
    )


def _import_runtime(package: Path, manifest: dict[str, Any]) -> Any:
    """The package's ``decision2`` runtime, copied next to this module and imported relative to it."""
    files = {
        name: digest
        for name, digest in (manifest.get("files_sha256") or {}).items()
        if name.startswith(RUNTIME_DIR) and name.endswith(".py")
    }
    if f"{RUNTIME_DIR}__init__.py" not in files:
        raise ValueError("The repository has no decision2/ runtime")
    if not __package__:
        raise ImportError(
            "Load this model with transformers: AutoModel.from_pretrained(repo, trust_remote_code=True)"
        )
    name = (
        "decision2_"
        + hashlib.sha256(json.dumps(sorted(files.items())).encode("utf-8")).hexdigest()[
            :16
        ]
    )
    home = Path(__file__).resolve().parent
    target = home / name
    with _LOCK:
        if not _runtime_matches(target, files):
            staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=home))
            try:
                for relative, digest in files.items():
                    copy = staging / relative[len(RUNTIME_DIR) :]
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(package / relative, copy)
                    if _sha256(copy) != digest:
                        raise ValueError(f"{relative} differs from MODEL_MANIFEST.json")
                if target.exists():
                    shutil.rmtree(target)
                try:
                    os.rename(staging, target)
                except OSError:
                    # Another process placed the same runtime first.
                    if not _runtime_matches(target, files):
                        raise
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        importlib.invalidate_caches()
        return importlib.import_module(f"{__package__}.{name}")


def _accepting(function: Any) -> Any:
    """``function`` called with only the keywords it takes, as Transformers' fallback wrapper calls it."""
    parameters = inspect.signature(function).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return function

    @functools.wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **{k: v for k, v in kwargs.items() if k in parameters})

    return call


def _cpu_reference_layers(root: torch.nn.Module) -> int:
    """Bind the Qwen3.5 gated-delta layers under ``root`` to the PyTorch reference functions.

    Transformers binds those functions at import to the flash-linear-attention / causal-conv1d kernels when
    they are installed, and the kernels are GPU-only. Each layer gets its own forward whose globals name the
    reference functions, which is what Transformers runs without the kernels; nothing global changes.
    Returns the number of layers rebound.
    """
    modeling = sys.modules.get(QWEN3_5_MODELING)
    layer_class = getattr(modeling, "Qwen3_5GatedDeltaNet", None)
    if layer_class is None or not any(name in sys.modules for name in KERNEL_PACKAGES):
        return 0
    references = {
        name: _accepting(inspect.unwrap(getattr(modeling, name)))
        for name in GATED_DELTA
        if callable(getattr(modeling, name, None))
    }
    forward = inspect.unwrap(layer_class.forward)
    reference_forward = types.FunctionType(
        forward.__code__,
        {**forward.__globals__, **references},
        forward.__name__,
        forward.__defaults__,
        forward.__closure__,
    )
    reference_forward.__kwdefaults__ = forward.__kwdefaults__
    layers = [m for m in root.modules() if isinstance(m, layer_class)]
    for layer in layers:
        layer.forward = types.MethodType(reference_forward, layer)
    return len(layers)


class Decision2Model(PreTrainedModel):
    """A Decision 2.0 package behind System One: ``system_one(state=..., questions={...})``."""

    config_class = Decision2Config
    base_model_prefix = "decision"
    main_input_name = "state"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, config: Decision2Config):
        super().__init__(config)
        self.runtime = None
        self.cpu_reference_layers = 0
        self._source: Path | None = None
        self._hub: dict[str, Any] = {}
        self._options: dict[str, Any] = {}
        self._base_path: str | None = None
        self.post_init()

    def _init_weights(self, module: Any) -> None:
        """Every weight comes from the verified package; nothing is initialized here."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *model_args: Any,
        config: Decision2Config | None = None,
        **kwargs: Any,
    ) -> Decision2Model:
        """Load a Hub repository or a local download through its native runtime.

        Hub options: ``revision``, ``cache_dir``, ``token``, ``local_files_only``, ``force_download``.
        ``device`` or ``device_map`` names one device (default: cuda:0 if a GPU is visible, else CPU).
        Runtime options pass through: ``threads``, ``bf16_resident``, ``graphs``, ``kernels``,
        ``share_context`` (the default for requests that do not set it; see ``system_one``) and, for a
        base-bound adapter, ``base_path`` (a local copy of the pinned base revision). Numerics are the
        runtime's, so ``dtype`` is only None or "auto".
        """
        if model_args:
            raise TypeError("Decision 2.0 models take no positional model arguments")
        hub = {k: kwargs.pop(k) for k in HUB_OPTIONS if k in kwargs}
        if kwargs.pop("subfolder", "") not in ("", None):
            raise ValueError("A Decision 2.0 package loads from the repository root")
        options = {k: kwargs.pop(k) for k in RUNTIME_OPTIONS if k in kwargs}
        device_map = kwargs.pop("device_map", None)
        for key in ("dtype", "torch_dtype"):
            if kwargs.pop(key, None) not in (None, "auto"):
                raise ValueError(
                    f"{key}: Decision 2.0 numerics are fixed by the runtime (FP32 on CPU; on a GPU BF16 "
                    "autocast with an FP32 head); pass None or 'auto'"
                )
        if kwargs.pop("attn_implementation", None) not in (None, "sdpa"):
            raise ValueError("Decision 2.0 backbones use SDPA attention")
        loading_info = kwargs.pop("output_loading_info", False)
        for key in LOADER_FLAGS:
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(
                f"Unsupported keyword arguments for a Decision 2.0 model: {sorted(kwargs)}"
            )
        options["device"] = _device(options.get("device"), device_map)
        if config is None:
            config = Decision2Config.from_pretrained(
                pretrained_model_name_or_path,
                **{k: v for k, v in hub.items() if v is not None},
            )
        model = cls(config)
        model._source = _package_dir(pretrained_model_name_or_path, config, hub)
        model._hub = hub
        model._options = {k: v for k, v in options.items() if v is not None}
        model._load(model._options)
        model.name_or_path = str(pretrained_model_name_or_path)
        if loading_info:
            return model, {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        return model

    def _load(self, options: dict[str, Any]) -> None:
        view, temporary = _link_free(self._source)
        try:
            manifest = json.loads(
                (view / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
            )
            runtime = _import_runtime(view, manifest)
            options = dict(options)
            if (
                manifest.get("profile") == "qwen-adapter"
                and options.get("base_path") is None
            ):
                if self._base_path is None:
                    self._base_path = _pinned_base(manifest["base"], self._hub)
                options["base_path"] = self._base_path
            loader = runtime.Decision2.from_pretrained
            loaded = loader(
                view, **_supported(loader, options, "This repository's runtime")
            )
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
        module = getattr(loaded.backend, "model", None)
        self._modules.pop("decision", None)
        if isinstance(module, torch.nn.Module):
            self.decision = module
        self.runtime = loaded
        self.cpu_reference_layers = (
            _cpu_reference_layers(module)
            if isinstance(module, torch.nn.Module)
            and self._runtime_device().type == "cpu"
            else 0
        )
        super().train(False)

    def _require(self) -> Any:
        if self.runtime is None:
            raise RuntimeError("Load the model with from_pretrained")
        return self.runtime

    def _runtime_device(self) -> torch.device:
        return torch.device(getattr(self._require().backend, "device", "cpu"))

    @property
    def model_name(self) -> str:
        return self._require().model_name

    @property
    def max_input_tokens(self) -> int:
        return self._require().max_input_tokens

    @property
    def manifest(self) -> dict[str, Any]:
        return self._require().manifest

    def system_one(
        self, *, state: Any, questions: dict[str, Any], share_context: Any = None
    ) -> dict[str, Any]:
        """Typed Choice / Noul / Score answers about one state: ``{"model", "answers", "usage"}``.

        ``questions`` maps question IDs to ``{"type": "choice" | "noul" | "score", "instructions": ...,
        "criteria": ...}``; over-budget input is answered with ``max_length_exceeded``, never truncated.
        ``share_context=True`` computes the input the questions share once instead of once per question
        (much faster for many questions; near-tie answers can differ slightly from the exact path);
        None keeps the default given to ``from_pretrained`` (off unless set there).
        """
        if share_context is None:
            return self._require().system_one(state=state, questions=questions)
        return self._require().system_one(
            state=state, questions=questions, share_context=share_context
        )

    def forward(
        self,
        state: Any = None,
        questions: dict[str, Any] | None = None,
        share_context: Any = None,
    ) -> dict[str, Any]:
        return self.system_one(
            state=state, questions=questions, share_context=share_context
        )

    def to(self, *args: Any, **kwargs: Any) -> Decision2Model:
        """Move to another device by loading the package there through the native runtime."""
        device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None or memory_format is not None:
            raise TypeError(
                "Decision 2.0 numerics are fixed by the runtime; only the device can change"
            )
        if device is None or _same_device(device, self._runtime_device()):
            return self
        with _LOCK:
            self._load({**self._options, "device": str(device)})
            self._options["device"] = str(device)
        return self

    def cuda(self, device: Any = None) -> Decision2Model:
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return self.to(device if device is not None else "cuda")

    def cpu(self) -> Decision2Model:
        return self.to("cpu")

    def _cast(self, *args: Any, **kwargs: Any) -> Decision2Model:
        raise TypeError(
            "Decision 2.0 numerics are fixed by the runtime; dtype casts are not supported"
        )

    def half(self, *args: Any, **kwargs: Any) -> Decision2Model:
        return self._cast()

    def float(self, *args: Any, **kwargs: Any) -> Decision2Model:
        return self._cast()

    def bfloat16(self, *args: Any, **kwargs: Any) -> Decision2Model:
        return self._cast()

    def double(self, *args: Any, **kwargs: Any) -> Decision2Model:
        return self._cast()

    def train(self, mode: bool = True) -> Decision2Model:
        if mode:
            raise RuntimeError("Decision 2.0 models are inference-only")
        return super().train(False)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The repository itself is the package; copy it with "
            "huggingface_hub.snapshot_download(repo_id, local_dir=...)"
        )

    def push_to_hub(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Decision 2.0 packages are published by their release pipeline"
        )
