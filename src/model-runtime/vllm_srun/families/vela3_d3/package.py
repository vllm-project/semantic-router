"""The Vela 3.0 d3-size package (``vela3/3``): what it holds and how its bytes are verified.

vela3-lite, -mini, -flash and -turbo hold no backbone: ``vela3_config.json``
pins a Decision 3.0 package (``base``: repository and 40-hex revision) whose
weights run unchanged, and names the gated LoRA adapters over its text model
(``adapters``: file, rank, alpha, module count, first layer), the span heads
(``span.head_file``, ``span.ner.head_file``, ``span.relation.head_file``),
the router questions and the label definition cache. An adapter file holds
FP32 factors ``layers.<i>.<module>.A`` (rank x in) and ``.B`` (out x rank)
and names its rank, alpha and first layer in its metadata.

Every file the family loads must match the built-in table's digests for a
pinned revision, whose download holds only those files, and otherwise the
package's ``SHA256SUMS``; bundled Python is never read.
Image inputs are out of scope, so the image-halu adapter and head are neither
loaded nor checked. The identity is the SHA-256 of the loaded weight files'
digests and the base's identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import (
    named_files,
    read_json,
    safetensors_header,
    sha256_json,
)
from ...registry.resolve import REVISION
from ..vela3.package import CONFIG_FILE, SUMS_FILE, package_file, read_sums

FORMAT = "vela3/3"
# Read for images only (a non-goal here): never loaded.
IMAGE_ADAPTERS = frozenset({"image_halu"})
# The span heads by engine name and where vela3_config.json names their files.
HEAD_FIELDS = {"span": (), "ner": ("ner",), "relation": ("relation",)}
MATRIX = 2  # a LoRA factor's dimensions


def member_of(root: Path) -> bool:
    """Whether a package directory holds a ``vela3/3`` package, read from its ``vela3_config.json`` only."""
    try:
        config = json.loads((Path(root) / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and config.get("format") == FORMAT


@dataclass(frozen=True)
class Base:
    """The pinned Decision 3.0 package the adapters were trained on."""

    repo_id: str
    revision: str


@dataclass(frozen=True)
class Adapter:
    """One gated LoRA adapter: its file, rank, alpha, first layer and the modules it wraps."""

    name: str
    file: str
    rank: int
    alpha: float
    first: int
    modules: tuple[str, ...]

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass(frozen=True)
class Vela3D3Package:
    """A verified ``vela3/3`` package; ``base_package`` is its verified Decision 3.0 base."""

    root: Path
    vela3: dict[str, Any]
    config: dict[str, Any]
    base: Base
    adapters: dict[str, Adapter]
    heads: dict[str, str]
    router_question_ids: frozenset[str]
    router_questions: dict[str, Any]
    files: dict[str, str]
    weights_sha256: str
    base_package: Any = None
    model_sha256: str = ""


def _base(vela3: dict[str, Any]) -> Base:
    base = vela3.get("base")
    if not isinstance(base, dict) or not isinstance(base.get("repo_id"), str):
        raise PackageError(f"{CONFIG_FILE}: base must name the Decision 3.0 repository")
    revision = str(base.get("revision"))
    if not REVISION.fullmatch(revision):
        raise PackageError(f"{CONFIG_FILE}: base revision must be a 40-hex commit")
    return Base(base["repo_id"], revision)


def _heads(vela3: dict[str, Any]) -> dict[str, str]:
    span = vela3.get("span")
    if not isinstance(span, dict):
        raise PackageError(f"{CONFIG_FILE}: span must describe the span heads")
    heads = {"span": package_file(span.get("head_file"), "span.head_file")}
    for name, path in HEAD_FIELDS.items():
        if not path:
            continue
        entry = span.get(path[0])
        if isinstance(entry, dict) and entry.get("head_file"):
            heads[name] = package_file(entry["head_file"], f"span.{path[0]}.head_file")
    return heads


def adapter_of(root: Path, name: str, spec: Any) -> Adapter:
    """Read one adapter's header and check it against its ``vela3_config.json`` entry (no tensor is read)."""
    if not isinstance(spec, dict):
        raise PackageError(f"{CONFIG_FILE}: adapters.{name} must be an object")
    file = package_file(spec.get("file"), f"adapters.{name}.file")
    header = safetensors_header(root / file)
    metadata = header.pop("__metadata__", None) or {}
    try:
        rank, alpha = int(metadata["r"]), float(metadata["alpha"])
        first = int(metadata.get("first", 0))
    except (KeyError, ValueError) as exc:
        raise PackageError(f"adapter {name}: {file} names no rank and alpha") from exc
    factors: dict[str, set[str]] = {}
    for key in header:
        module, _, factor = key.rpartition(".")
        if factor not in ("A", "B") or not module.startswith("layers."):
            raise PackageError(f"adapter {name}: unexpected tensor {key}")
        factors.setdefault(module, set()).add(factor)
    if any(found != {"A", "B"} for found in factors.values()):
        raise PackageError(f"adapter {name}: unpaired A / B tensors")
    modules = tuple(sorted(factors))
    declared = (spec.get("r"), spec.get("alpha"), spec.get("modules"))
    if (rank, alpha, len(modules)) != (declared[0], declared[1], declared[2]):
        raise PackageError(
            f"adapter {name} does not match {CONFIG_FILE}: rank {rank}, alpha {alpha:g}, "
            f"{len(modules)} modules"
        )
    if spec.get("first_layer") is not None and int(spec["first_layer"]) != first:
        raise PackageError(f"adapter {name}: first layer differs from {CONFIG_FILE}")
    for module in modules:
        down = list(header[f"{module}.A"].get("shape") or ())
        up = list(header[f"{module}.B"].get("shape") or ())
        if len(down) != MATRIX or len(up) != MATRIX or down[0] != rank or up[1] != rank:
            raise PackageError(
                f"adapter {name}: factor shapes at {module} do not fit rank {rank}"
            )
    return Adapter(name, file, rank, alpha, first, modules)


def verify(root: Path, expected: dict[str, str] | None) -> Vela3D3Package:
    """Check every loaded file; ``expected`` is the built-in table's digests for a pinned revision."""
    root = Path(root)
    if not member_of(root):
        raise PackageError(f"{CONFIG_FILE} does not describe a {FORMAT} package")
    vela3 = read_json(root / CONFIG_FILE, mapping=True)
    base = _base(vela3)
    specs = vela3.get("adapters")
    if not isinstance(specs, dict) or "decision" not in specs or "span" not in specs:
        raise PackageError(
            f"{CONFIG_FILE}: adapters must name the decision and span adapters"
        )
    heads = _heads(vela3)
    questions_file = package_file(
        vela3.get("router_questions_file"), "router_questions_file"
    )
    definitions = vela3.get("definitions") or {}
    data = ["config.json", CONFIG_FILE, questions_file]
    if definitions.get("cache_file"):
        data.append(package_file(definitions["cache_file"], "definitions.cache_file"))
    weights = [
        package_file((spec or {}).get("file"), f"adapters.{name}.file")
        for name, spec in specs.items()
        if name not in IMAGE_ADAPTERS
    ]
    weights += list(heads.values())
    if definitions.get("writer") == "grafted_head":
        weights.append(
            package_file(definitions.get("lm_head_file"), "definitions.lm_head_file")
        )
    weights = sorted(set(weights))
    files = named_files(root, [*data, *weights])
    if expected is not None:
        wrong = sorted(
            name
            for name in set(expected) | set(files)
            if expected.get(name) != files.get(name)
        )
        if wrong:
            raise PackageError(f"files differ from the pinned revision: {wrong[:5]}")
    else:
        sums = read_sums(root, required=True)
        mismatched = sorted(
            name for name, digest in files.items() if sums.get(name) != digest
        )
        if mismatched:
            raise PackageError(f"files differ from {SUMS_FILE}: {mismatched[:5]}")
    config = read_json(root / "config.json", mapping=True)
    if config.get("model_type") != "qwen3_5":
        raise PackageError("config.json does not describe a Qwen3.5 model")
    ids = vela3.get("router_question_ids")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise PackageError(f"{CONFIG_FILE}: router_question_ids must list question IDs")
    questions = read_json(root / questions_file, mapping=True).get("trained_questions")
    if not isinstance(questions, dict):
        raise PackageError(f"{questions_file} lists no trained router questions")
    adapters = {
        name: adapter_of(root, name, spec)
        for name, spec in specs.items()
        if name not in IMAGE_ADAPTERS
    }
    return Vela3D3Package(
        root=root,
        vela3=vela3,
        config=config,
        base=base,
        adapters=adapters,
        heads=heads,
        router_question_ids=frozenset(ids),
        router_questions=questions,
        files=files,
        weights_sha256=sha256_json({name: files[name] for name in weights}),
    )


def identity(package: Vela3D3Package, base_sha256: str) -> str:
    """The package's identity: its weight files and the base it runs on."""
    return sha256_json({"weights": package.weights_sha256, "base": base_sha256})
