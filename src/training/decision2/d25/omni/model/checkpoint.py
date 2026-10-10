"""Code-readout v1 checkpoint I/O for Decision 2.5 Omni.

A checkpoint directory holds a ``Qwen3_5Model`` backbone (``config.json``, ``model-*.safetensors``
and ``model.safetensors.index.json``; tensors named ``visual.*`` and ``language_model.*``),
``readout.safetensors`` (``{"weight": [255, hidden]}``), ``decision_config.json`` and the
tokenizer, processor and chat-template files. This is the Vega layout (vega/SPEC.md) with the
vision tower always present.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
STOCK_SHARDS_SHA256 = {
    "model-00001-of-00018.safetensors": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
    "model-00002-of-00018.safetensors": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
    "model-00003-of-00018.safetensors": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
    "model-00004-of-00018.safetensors": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
    "model-00005-of-00018.safetensors": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
    "model-00006-of-00018.safetensors": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
    "model-00007-of-00018.safetensors": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
    "model-00008-of-00018.safetensors": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
    "model-00009-of-00018.safetensors": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
    "model-00010-of-00018.safetensors": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
    "model-00011-of-00018.safetensors": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
    "model-00012-of-00018.safetensors": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
    "model-00013-of-00018.safetensors": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
    "model-00014-of-00018.safetensors": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
    "model-00015-of-00018.safetensors": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
    "model-00016-of-00018.safetensors": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
    "model-00017-of-00018.safetensors": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
    "model-00018-of-00018.safetensors": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
}
STOCK_FILES_SHA256 = {
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
}
STOCK_FILES_GIT_OID = {
    "chat_template.jinja": "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3",
    "config.json": "706cebd746c4b6f2b1d1f892630867acfdfd3df8",
    "merges.txt": "a494e019ca1502219fd0128658b979e5f05ae8e8",
    "model.safetensors.index.json": "da35e3c564457dface7d138f0b6cac284ff8958c",
    "preprocessor_config.json": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
    "tokenizer_config.json": "5de744b3fca2129d7186979ae47c06be33903243",
    "video_preprocessor_config.json": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
    "vocab.json": "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
}
STOCK_VISION_PARAMETERS = 460_730_096
NUM_CODES = 255

DECISION_CONFIG = "decision_config.json"
READOUT_FILE = "readout.safetensors"
INDEX_FILE = "model.safetensors.index.json"
PROCESSOR_FILES = (
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
WRAPPER_PREFIXES = ("_orig_mod.", "_fsdp_wrapped_module.", "module.", "backbone.")
TEXT_ROOTS = ("embed_tokens.", "layers.", "norm.")


def canonical_name(key: str) -> str | None:
    """Map a tensor name of a known layout to ``Qwen3_5Model`` naming; None drops it.

    Known layouts: ``Qwen3_5Model`` (``visual.*``, ``language_model.*``),
    ``Qwen3_5ForConditionalGeneration`` (``model.``-prefixed, plus ``lm_head`` and ``mtp``), a bare
    ``Qwen3_5TextModel`` (``embed_tokens``, ``layers``, ``norm``), and wrapper prefixes left by
    trainers. ``readout.weight`` is returned unchanged.
    """
    stripped = True
    while stripped:
        stripped = False
        for prefix in WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key, stripped = key[len(prefix) :], True
    if key.startswith(("lm_head.", "mtp.")):
        return None
    if key.startswith("model."):
        key = key[len("model.") :]
    if key.startswith(("visual.", "language_model.")) or key == "readout.weight":
        return key
    if key.startswith(TEXT_ROOTS):
        return "language_model." + key
    raise ValueError(f"unrecognised checkpoint tensor {key!r}")


def component(name: str) -> str:
    """``vision_encoder``, ``vision_merger``, ``language`` or ``readout`` for a canonical name."""
    if name.startswith("visual.merger."):
        return "vision_merger"
    if name.startswith("visual."):
        return "vision_encoder"
    if name.startswith("language_model."):
        return "language"
    if name == "readout.weight":
        return "readout"
    raise ValueError(f"not a canonical tensor name: {name!r}")


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_blob_oid(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def tensor_digest(tensor: torch.Tensor) -> str:
    """sha256 over dtype, shape and raw bytes: equal digests mean bit-equal tensors."""
    digest = hashlib.sha256(f"{tensor.dtype}|{tuple(tensor.shape)}|".encode())
    digest.update(tensor_bytes(tensor))
    return digest.hexdigest()


def digest_of_digests(digests: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def weight_files(directory: str | Path) -> dict[str, str]:
    """Tensor name -> safetensors file name of a checkpoint directory (sharded or single file)."""
    root = Path(directory)
    index = root / INDEX_FILE
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        for name in set(weight_map.values()):
            if Path(name).name != name or not (root / name).exists():
                raise FileNotFoundError(f"{root}: missing shard {name}")
        return dict(weight_map)
    single = root / "model.safetensors"
    if single.exists():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt") as handle:
            return {key: single.name for key in handle.keys()}
    raise FileNotFoundError(f"{root}: no model.safetensors or {INDEX_FILE}")


def iter_tensors(
    directory: str | Path, names: set[str] | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream ``(stored name, tensor)`` shard by shard, optionally restricted to ``names``."""
    from safetensors import safe_open

    root = Path(directory)
    by_file: dict[str, list[str]] = {}
    for key, name in weight_files(root).items():
        if names is None or key in names:
            by_file.setdefault(name, []).append(key)
    for name in sorted(by_file):
        with safe_open(str(root / name), framework="pt") as handle:
            for key in sorted(by_file[name]):
                yield key, handle.get_tensor(key)


def read_decision_config(directory: str | Path) -> dict[str, Any]:
    config = json.loads((Path(directory) / DECISION_CONFIG).read_text())
    if config.get("format_version") != 1:
        raise ValueError(f"{directory}: unsupported decision_config format_version")
    codes, token_ids = config.get("codes"), config.get("token_ids")
    if (
        not isinstance(codes, list)
        or not isinstance(token_ids, list)
        or len(codes) != len(token_ids)
    ):
        raise ValueError(
            f"{directory}: decision_config needs matching codes and token_ids"
        )
    return config


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, target)


def load_readout(directory: str | Path) -> torch.Tensor:
    from safetensors.torch import load_file

    weight = load_file(str(Path(directory) / READOUT_FILE))["weight"]
    if weight.ndim != 2 or weight.shape[0] != NUM_CODES:
        raise ValueError(
            f"{directory}: readout must be [{NUM_CODES}, hidden], got {tuple(weight.shape)}"
        )
    return weight


def save_readout(directory: str | Path, weight: torch.Tensor) -> str:
    from safetensors.torch import save_file

    if weight.ndim != 2 or weight.shape[0] != NUM_CODES:
        raise ValueError(
            f"readout must be [{NUM_CODES}, hidden], got {tuple(weight.shape)}"
        )
    path = Path(directory) / READOUT_FILE
    save_file({"weight": weight.detach().cpu().contiguous()}, str(path))
    return file_sha256(path)


class ShardWriter:
    """Size-capped ``model-XXXXX-of-YYYYY.safetensors`` shards plus the Transformers index."""

    def __init__(
        self, directory: str | Path, max_shard_bytes: int = 5 * 1024**3
    ) -> None:
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = max_shard_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.parts: list[tuple[Path, list[str]]] = []
        self.total_bytes = 0
        self.digests: dict[str, str] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.digests or name in self.pending:
            raise ValueError(f"duplicate tensor {name}")
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + size > self.max_shard_bytes:
            self._flush()
        self.pending[name] = tensor.detach().cpu().contiguous()
        self.pending_bytes += size
        self.total_bytes += size

    def _flush(self) -> None:
        from safetensors.torch import save_file

        path = self.root / f".part-{len(self.parts):05d}.safetensors"
        save_file(self.pending, str(path), metadata={"format": "pt"})
        self.parts.append((path, sorted(self.pending)))
        for name, tensor in self.pending.items():
            self.digests[name] = tensor_digest(tensor)
        self.pending, self.pending_bytes = {}, 0

    def close(self) -> dict[str, str]:
        """Finish the shards; returns shard file name -> sha256."""
        if self.pending:
            self._flush()
        if not self.parts:
            raise ValueError("no tensors written")
        count = len(self.parts)
        weight_map: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for number, (path, names) in enumerate(self.parts, start=1):
            final = self.root / f"model-{number:05d}-of-{count:05d}.safetensors"
            os.replace(path, final)
            hashes[final.name] = file_sha256(final)
            weight_map.update({name: final.name for name in names})
        write_json(
            self.root / INDEX_FILE,
            {
                "metadata": {"total_size": self.total_bytes},
                "weight_map": dict(sorted(weight_map.items())),
            },
        )
        return hashes


def expected_backbone_shapes(config) -> dict[str, tuple[int, ...]]:
    """Persistent state-dict names and shapes of a ``Qwen3_5Model`` built from ``config``."""
    from transformers import Qwen3_5Model

    with torch.device("meta"):
        model = Qwen3_5Model(config)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def copy_processor_files(source: str | Path, destination: str | Path) -> dict[str, str]:
    """Copy the tokenizer/processor files present in ``source``; returns name -> sha256."""
    import shutil

    hashes: dict[str, str] = {}
    for name in PROCESSOR_FILES:
        path = Path(source) / name
        if path.exists():
            shutil.copyfile(path, Path(destination) / name)
            hashes[name] = file_sha256(path)
    return hashes
