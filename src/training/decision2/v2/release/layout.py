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

from v2.release.automap import CONFIG_FIELDS

PACKAGE_SCHEMA = "dev2-package/1"
MANIFEST_SCHEMA = "dev2-package-manifest/1"
MANIFEST_NAME = "MODEL_MANIFEST.json"
POINTER_NAME = "config.json"
POINTER = {"decision_format": "vllm-sr-decision", "format_version": 2}
RUNTIME_DIR = "decision2"
VENDOR_DIR = "decision2/_vendor"
ORG = "llm-semantic-router"
TIERS = {"0.6B": 0.6e9, "0.8B": 0.8e9, "2B": 2e9, "4B": 4e9, "9B": 9e9, "27B": 27e9}
# The codename follows the size slot across generations (user naming decision 2026-10-02 00:05 UTC+8).
CODENAMES = {
    "0.6B": "Kai",
    "0.8B": "Eos",
    "2B": "Sol",
    "4B": "Nox",
    "9B": "Lux",
    "27B": "Vega",
}
NAME_PREFIX = "Decision-2.0"
MODEL_NAME = re.compile(
    r"Decision-2\.0-(?P<codename>Kai|Eos|Sol|Nox|Lux|Vega)-(?P<size>0\.[1-9]|[1-9][0-9]?)B\Z"
)
RELEASE_REPO = re.compile(
    r"llm-semantic-router/Decision-2\.0-(?:Kai|Eos|Sol|Nox|Lux|Vega)-(?:0\.[1-9]|[1-9][0-9]?)B\Z"
)
STAGING_REPO = re.compile(
    r"llm-semantic-router/dev2-release-staging(?:-[a-z0-9]{1,24})?\Z"
)
# Repositories released before the rename (moved with HfApi.move_repo, so the old IDs redirect). Historical
# gate receipts and decisions keep these IDs; new releases never use them.
FORMER_REPOS = {
    f"{ORG}/DEV2.0-{tier}": f"{ORG}/{NAME_PREFIX}-{codename}-{tier}"
    for tier, codename in CODENAMES.items()
}
SAME_SIZE_RATIO = 1.25
# "tier": Decision-2.0-<codename>-<size tier>; "loaded-parameters": ...-<rounded loaded count> (brief section 2);
# "base": ...-<size label of the base model>, which must fall in the loaded count's tier. The codename is the tier's.
NAME_BASES = ("tier", "loaded-parameters", "base")
BASE_SIZE = re.compile(r"(?:^|[-_])([0-9]+(?:\.[0-9]+)?)B(?=$|[-_])")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
CARD_FILES = ("README.md", "LICENSE")
# JevArena overall and by type, then the Jev Decision Index against size and by area.
CHART_FILES = (
    "assets/jevarena.png",
    "assets/jevarena-types.png",
    "assets/index-pareto.png",
    "assets/index-areas.png",
)
CARD_ASSETS = ("assets/banner.png", *CHART_FILES)
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
MODEL_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt", ".bf16z"}
BF16Z_SUFFIX = ".safetensors.bf16z"


def is_weight(name: str) -> bool:
    """A safetensors weight file, stored plain or as bf16z (v2/release/runtime/bf16z.py)."""
    return name.endswith((".safetensors", BF16Z_SUFFIX))


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
    """Element count from the header only; weights are never read (nor decompressed)."""
    if str(path).endswith(BF16Z_SUFFIX):
        from v2.release.runtime.bf16z import original_header

        header = original_header(Path(path))
    else:
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


def tier_name(tier: str, size: str | None = None) -> str:
    """Decision-2.0-<codename of the tier>-<size label, default the tier>."""
    return f"{NAME_PREFIX}-{CODENAMES[tier]}-{size or tier}"


def name_for(parameters: int) -> str:
    tier = tier_for(parameters)
    if tier is None:
        raise ValueError(f"{parameters:,} parameters fall outside every size tier")
    return tier_name(tier)


def count_name(parameters: int) -> str:
    """Decision-2.0-<codename>-<loaded count>: one decimal below 1B, whole billions from 1B."""
    name_for(parameters)
    billions = parameters / 1e9
    size = f"{billions:.1f}" if billions < 1 else f"{round(billions)}"
    return tier_name(tier_for(parameters), f"{size}B")


def current_repo(repo_id: str) -> str:
    """The repository ID now, for an ID that may predate the rename."""
    return FORMER_REPOS.get(repo_id, repo_id)


def base_size_label(base_model: str) -> str:
    """Size label of a base model repository, e.g. ``Qwen/Qwen3.5-9B`` -> ``9B``."""
    labels = BASE_SIZE.findall(base_model.rsplit("/", 1)[-1])
    if len(labels) != 1:
        raise ValueError(f"{base_model} does not name exactly one size (<n>B)")
    return f"{labels[0]}B"


def base_name(parameters: int, base_model: str) -> str:
    """Decision-2.0-<codename>-<base model size>, if that size and the loaded count share a size tier."""
    label = base_size_label(base_model)
    tier = tier_for(parameters)
    if tier is None:
        raise ValueError(f"{parameters:,} parameters fall outside every size tier")
    if tier_for(round(float(label[:-1]) * 1e9)) != tier:
        raise ValueError(
            f"{base_model} ({label}) is outside the {tier} tier of {parameters:,} loaded parameters"
        )
    return tier_name(tier, label)


def release_name(
    parameters: int, basis: str = "tier", base_model: str | None = None
) -> str:
    if basis not in NAME_BASES:
        raise ValueError(f"name_basis is one of {NAME_BASES}")
    if basis == "base":
        if not base_model:
            raise ValueError("name_basis base needs the base model repository")
        return base_name(parameters, base_model)
    return (
        count_name(parameters) if basis == "loaded-parameters" else name_for(parameters)
    )


def check_repo(repo_id: str, model_name: str, *, staging: bool) -> None:
    if staging:
        if not STAGING_REPO.fullmatch(repo_id):
            raise ValueError(
                "Staging packages go only to llm-semantic-router/dev2-release-staging*"
            )
    elif repo_id in FORMER_REPOS:
        raise ValueError(f"{repo_id} was renamed; use {FORMER_REPOS[repo_id]}")
    elif not RELEASE_REPO.fullmatch(repo_id) or repo_id.rsplit("/", 1)[1] != model_name:
        raise ValueError("Release repositories are llm-semantic-router/<model name>")
    match = MODEL_NAME.fullmatch(model_name)
    tier = match and tier_for(round(float(match["size"]) * 1e9))
    if not match or tier is None or CODENAMES[tier] != match["codename"]:
        raise ValueError(
            "Model names are Decision-2.0-<codename>-<size>B with the size tier's codename ("
            + ", ".join(f"{t} {c}" for t, c in CODENAMES.items())
            + ")"
        )


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
    weights = [name for name in files if is_weight(name)]
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
    remote_code: bool = False,
) -> dict[str, Any]:
    """Root query file: a truthful map of the model files, read first at load.

    ``remote_code`` adds the 🤗 Transformers fields (``model_type``, ``auto_map``, ...) of the
    ``trust_remote_code`` modules the package ships at its root (``v2/release/automap``).
    """
    present = set(files)
    weights = sorted(n for n in files if is_weight(n))
    result: dict[str, Any] = {
        **(CONFIG_FIELDS if remote_code else {}),
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
    if any(n.endswith(BF16Z_SUFFIX) for n in files):
        result["weight_storage"] = {
            "codec": "bf16z/1",
            "restore": "decision2.bf16z.materialize (run by decision2.Decision2.from_pretrained)",
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
