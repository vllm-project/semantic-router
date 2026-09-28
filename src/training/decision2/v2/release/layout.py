"""One Hugging Face package layout for every Decision 2.0 size (stdlib only).

A package is the exact scored model bytes at their checkpoint-relative paths,
a small bundled System One runtime, a root ``config.json`` pointer (the file
the Hub's default download counter queries and the runtime reads first), a
``MODEL_MANIFEST.json`` inventory, and the product card. Profiles differ only
in which model files they carry; see ``README.md`` for the full tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PACKAGE_SCHEMA = "dev2-package/1"
MANIFEST_SCHEMA = "dev2-package-manifest/1"
MANIFEST_NAME = "MODEL_MANIFEST.json"
POINTER_NAME = "config.json"
POINTER = {"decision_format": "vllm-sr-decision", "format_version": 2}
RUNTIME_DIR = "decision2"
VENDOR_DIR = "decision2/_vendor"
ORG = "llm-semantic-router"
RELEASE_REPO = re.compile(r"llm-semantic-router/DEV2\.0-(0\.6|0\.8|2|4|9|27)B\Z")
STAGING_REPO = re.compile(
    r"llm-semantic-router/dev2-release-staging(?:-[a-z0-9]{1,24})?\Z"
)
MODEL_NAME = re.compile(r"DEV2\.0-(0\.6|0\.8|2|4|9|27)B\Z")
TIERS = {"0.6B": 0.6e9, "0.8B": 0.8e9, "2B": 2e9, "4B": 4e9, "9B": 9e9, "27B": 27e9}
SAME_SIZE_RATIO = 1.25
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
CARD_FILES = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "ATTRIBUTIONS.md",
    "evaluation/EVALUATION.md",
    "evaluation/manifest.json",
)
CHART_FILES = (
    "assets/jevarena-v3-rank.svg",
    "assets/jevarena-v3-model-task.svg",
    "assets/jevbench-public231-rank.svg",
)
# Hub-side files that a real download may add; never part of the package.
HUB_ADDED = (".gitattributes",)
TRAINING_ONLY = {"trainer_state.pt", "checkpoint.json", "optimizer.pt", "scheduler.pt"}
TOKENIZER_ROOT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
)
MODEL_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt"}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_json(path: Path, value: Any) -> str:
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("xb") as target:
        target.write(data)
    return hashlib.sha256(data).hexdigest()


def safe_relative(name: str) -> bool:
    relative = PurePosixPath(name)
    return (
        bool(name)
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == name
        and "\\" not in name
    )


def transient(relative: PurePosixPath) -> bool:
    """Bytecode and HF local-dir metadata are ignored; everything else counts."""
    if relative.parts[:2] == (".cache", "huggingface"):
        return True
    return "__pycache__" in relative.parts and relative.suffix == ".pyc"


def inventory(root: Path, *, allow_hub_added: bool = False) -> dict[str, str]:
    """SHA-256 of every regular file; symlinks and unsafe names are refused."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Package root is absent or linked: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise ValueError(f"Package contains a link: {relative}")
        if path.is_dir() or transient(relative):
            continue
        name = relative.as_posix()
        if not path.is_file() or not safe_relative(name):
            raise ValueError(f"Unsafe package entry: {name}")
        if allow_hub_added and name in HUB_ADDED:
            continue
        files[name] = sha_file(path)
    return files


def safetensors_count(path: Path) -> int:
    """Element count from the header only; weights are never read."""
    with Path(path).open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        if not 2 <= size <= 256 << 20:
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path}")
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape") if isinstance(meta, dict) else None
        if not isinstance(shape, list) or any(
            type(dim) is not int or dim < 0 for dim in shape
        ):
            raise ValueError(f"Invalid tensor shape in {path}: {name}")
        total += math.prod(shape)
    return total


def safetensors_metadata(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(size))
    return header.get("__metadata__") or {}


def tier_for(parameters: int) -> str | None:
    """Nearest tier centre in log space, if within the frozen same-size ratio."""
    if type(parameters) is not int or parameters < 1:
        raise ValueError("Parameter count must be a positive integer")
    tier, centre = min(
        TIERS.items(), key=lambda item: abs(math.log(parameters / item[1]))
    )
    ratio = max(parameters / centre, centre / parameters)
    return tier if ratio <= SAME_SIZE_RATIO else None


def name_for(parameters: int) -> str:
    tier = tier_for(parameters)
    if tier is None:
        raise ValueError(f"{parameters:,} parameters fall outside every size tier")
    return f"DEV2.0-{tier}"


def check_repo(repo_id: str, model_name: str, *, staging: bool) -> None:
    if staging:
        if not STAGING_REPO.fullmatch(repo_id):
            raise ValueError(
                "Staging packages go only to llm-semantic-router/dev2-release-staging*"
            )
    elif not RELEASE_REPO.fullmatch(repo_id) or repo_id.rsplit("/", 1)[1] != model_name:
        raise ValueError("Release repositories are llm-semantic-router/<model name>")
    if not MODEL_NAME.fullmatch(model_name):
        raise ValueError("Model names are DEV2.0-<tier>")


@dataclass(frozen=True)
class Profile:
    """Which checkpoint files a package carries and where they live in it.

    ``prefix`` is prepended to checkpoint-relative names. Exports that verify
    their own directory manifest keep it intact under that subdirectory.
    """

    name: str
    runtime_family: str
    prefix: str
    description: str


PROFILES = {
    profile.name: profile
    for profile in (
        Profile(
            "kai-native",
            "decision2-kai-native",
            "native/",
            "Kai/Lex three-path bidirectional encoder: the exact native export "
            "tree, verified by its own MANIFEST.json, with the runtime vendored "
            "from the pinned Kai runtime revision.",
        ),
        Profile(
            "qwen-full",
            "decision2-qwen-full",
            "",
            "Self-contained Qwen3/Qwen3.5 text backbone (a standard Transformers "
            "backbone/ directory) plus the native candidate head.",
        ),
        Profile(
            "qwen-adapter",
            "decision2-qwen-adapter",
            "",
            "Base-bound PEFT LoRA (a standard adapter/ directory) plus the native "
            "candidate head; the base is pinned by repository revision and "
            "per-file SHA-256 and fetched or supplied at load time.",
        ),
        Profile(
            "encoder-marker",
            "decision2-encoder-marker",
            "encoder/",
            "Single bidirectional encoder with Kai-style candidate markers and "
            "the shared candidate head (0.6B encoder exports).",
        ),
    )
}


def package_path(profile: str, name: str) -> str:
    return PROFILES[profile].prefix + name


def _tree(root: Path, folder: str, suffixes: set[str] | None = None) -> list[str]:
    base = root / folder
    if not base.is_dir():
        return []
    names = []
    for path in sorted(base.rglob("*")):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise ValueError(f"Checkpoint contains a link: {relative}")
        if path.is_dir() or transient(relative):
            continue
        if suffixes is not None and relative.suffix not in suffixes:
            continue
        names.append(relative.as_posix())
    return names


def select_model_files(profile: str, checkpoint: Path) -> list[str]:
    """Checkpoint-relative model files, exactly the scored inference inputs."""
    checkpoint = Path(checkpoint)
    if profile in ("kai-native", "encoder-marker"):
        manifest = json.loads((checkpoint / "MANIFEST.json").read_text("utf-8"))
        return sorted([*manifest["files"], "MANIFEST.json"])
    if profile in ("qwen-full", "qwen-adapter"):
        root = [
            name
            for name in (
                "decision_config.json",
                "decision_head.safetensors",
                "dec_residual.safetensors",
            )
            + TOKENIZER_ROOT_FILES
            if (checkpoint / name).is_file()
        ]
        folder = "backbone" if profile == "qwen-full" else "adapter"
        # checkpoint_fingerprint covers exactly these suffixes under the folder.
        return sorted(root + _tree(checkpoint, folder, MODEL_SUFFIXES))
    raise ValueError(f"Unknown package profile: {profile}")


def parameter_files(profile: str, files: list[str]) -> dict[str, list[str]]:
    """Package weight paths grouped by component for header-based counting."""
    weights = [name for name in files if name.endswith(".safetensors")]
    if profile == "kai-native":
        return {"native": weights}
    if profile == "encoder-marker":
        return {
            "backbone": [n for n in weights if n.startswith("encoder/backbone/")],
            "head": [n for n in weights if n == "encoder/head.safetensors"],
        }
    groups = {
        "head": [n for n in weights if n == "decision_head.safetensors"],
        "residual": [n for n in weights if n == "dec_residual.safetensors"],
    }
    if profile == "qwen-full":
        return {"backbone": [n for n in weights if n.startswith("backbone/")], **groups}
    if profile == "qwen-adapter":
        return {"adapter": [n for n in weights if n.startswith("adapter/")], **groups}
    raise ValueError(f"Unknown package profile: {profile}")


def pointer(
    profile: str,
    model_name: str,
    files: list[str],
    *,
    calibration: str | None,
    base: dict[str, Any] | None,
    max_input_tokens: int,
) -> dict[str, Any]:
    """Root query file: a truthful map of the model files, read first at load."""
    present = set(files)
    weights = sorted(n for n in files if n.endswith(".safetensors"))
    result: dict[str, Any] = {
        **POINTER,
        "model_name": model_name,
        "runtime_family": PROFILES[profile].runtime_family,
        "package_schema": PACKAGE_SCHEMA,
        "manifest": MANIFEST_NAME,
        "runtime": {
            "python_package": RUNTIME_DIR,
            "entry": "decision2.Decision2.from_pretrained",
        },
        "max_input_tokens": max_input_tokens,
        "calibration": {"temperature_file": calibration} if calibration else None,
    }
    if profile == "kai-native":
        result.update(
            model_config="native/decision_config.json",
            backbone={
                "config": "native/encoder/config.json",
                "weights": ["native/encoder/model.safetensors"],
            },
            tokenizer={
                key: f"native/tokenizer/{name}"
                for key, name in (
                    ("json", "tokenizer.json"),
                    ("config", "tokenizer_config.json"),
                    ("special_tokens_map", "special_tokens_map.json"),
                )
                if f"native/tokenizer/{name}" in present
            },
            decision_weights={
                PurePosixPath(n).stem: n
                for n in weights
                if not n.startswith("native/encoder/")
            },
        )
        return result
    tokenizer = {
        key: name
        for key, name in (
            ("json", "tokenizer.json"),
            ("config", "tokenizer_config.json"),
            ("special_tokens_map", "special_tokens_map.json"),
            ("chat_template", "chat_template.jinja"),
        )
        if name in present
    }
    heads = {
        PurePosixPath(n).stem: n
        for n in (
            "decision_head.safetensors",
            "dec_residual.safetensors",
            "head.safetensors",
        )
        if n in present
    }
    if profile == "qwen-full":
        result.update(
            model_config="decision_config.json",
            backbone={
                "config": "backbone/config.json",
                "weights": [n for n in weights if n.startswith("backbone/")],
                **(
                    {"index": "backbone/model.safetensors.index.json"}
                    if "backbone/model.safetensors.index.json" in present
                    else {}
                ),
            },
            tokenizer=tokenizer,
            decision_weights=heads,
        )
    elif profile == "qwen-adapter":
        if not base:
            raise ValueError("Adapter packages need a pinned external base")
        result.update(
            model_config="decision_config.json",
            backbone={"repository": base["repo_id"], "revision": base["revision"]},
            adapter={
                "config": "adapter/adapter_config.json",
                "weights": [n for n in weights if n.startswith("adapter/")],
            },
            tokenizer=tokenizer,
            decision_weights=heads,
        )
    elif profile == "encoder-marker":
        result.update(
            model_config="encoder/decision_config.json",
            backbone={
                "config": "encoder/backbone/config.json",
                "weights": [n for n in weights if n.startswith("encoder/backbone/")],
            },
            tokenizer={
                key: f"encoder/tokenizer/{name}"
                for key, name in (
                    ("json", "tokenizer.json"),
                    ("config", "tokenizer_config.json"),
                    ("special_tokens_map", "special_tokens_map.json"),
                )
                if f"encoder/tokenizer/{name}" in present
            },
            decision_weights={"head": "encoder/head.safetensors"},
        )
    else:
        raise ValueError(f"Unknown package profile: {profile}")
    return result
