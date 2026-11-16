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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, TypedDict, TypeVar

from ...errors import PackageError
from ...registry.artifacts import (
    SUMS_FILE,
    named_files,
    read_json,
    read_sums,
    safetensors_header,
    sha256_json,
)
from ...registry.resolve import REVISION
from ..vela3_shared import CONFIG_FILE, config_file

FORMAT = "vela3/3"
# Read for images only (a non-goal here): never loaded.
IMAGE_ADAPTERS = frozenset({"image_halu"})
# A route names the adapter it runs and the head that reads it.
Route = Literal["span", "ner", "relation"]
# The most target tokens a span question reads, in about 21 windows: the vela3 family's default
# scan budget on a CPU (4 x 8,192). It bounds one question's forwards on every size, whatever the
# base's question cap (vela3-lite, -mini and -flash's bases allow 262,144 tokens).
SPAN_TARGET_TOKENS = 32768

T = TypeVar("T")


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
    """One gated LoRA adapter: its file, rank and alpha."""

    name: str
    file: str
    rank: int
    alpha: float


class Schema(TypedDict):
    """``span.schemas.<label set>``: its trained question and its labels' descriptions."""

    question: str
    labels: dict[str, str]


@dataclass(frozen=True)
class Evidence:
    """``span.evidence``: the evidence labels (lower case), question, threshold and decoding."""

    labels: frozenset[str]
    question: str
    threshold: float
    decode: dict[str, Any]


@dataclass(frozen=True)
class Ner:
    """``span.ner``: its head, its thresholds with and without label descriptions, and where name-only labels run."""

    head_file: str
    threshold: float
    threshold_names_only: float
    keeps_names_only: bool


@dataclass(frozen=True)
class Relation:
    """``span.relation``: its head, its threshold for one relation (``threshold_dev_pair``) and for several (``_multi``)."""

    head_file: str
    threshold_pair: float
    threshold_multi: float


@dataclass(frozen=True)
class SpanConfig:
    """A package's ``span`` section: its heads, label sets, schemas, questions and thresholds."""

    head_file: str
    router_label_sets: dict[str, frozenset[str]]
    schemas: dict[str, Schema]
    default_question: str
    threshold: float
    threshold_names_only: float
    thresholds_by_set: dict[str, float]
    pii_buckets: tuple[tuple[float, float], ...]
    evidence: Evidence | None
    ner: Ner | None
    relation: Relation | None

    @property
    def heads(self) -> dict[Route, str]:
        heads: dict[Route, str] = {"span": self.head_file}
        if self.ner is not None:
            heads["ner"] = self.ner.head_file
        if self.relation is not None:
            heads["relation"] = self.relation.head_file
        return heads

    @classmethod
    def parse(cls, value: Any) -> SpanConfig:
        """The ``span`` section, or a ``PackageError`` naming what is missing or malformed."""
        span = _Section.of(value, "span")
        sets = span.section("router_label_sets")
        schemas = span.section("schemas")
        by_set = span.section("thresholds_by_set")
        buckets = tuple(
            (bucket.number("max_tokens"), bucket.number("threshold"))
            for bucket in span.sections("pii_buckets")
        )
        if not any(limit >= SPAN_TARGET_TOKENS for limit, _ in buckets):
            raise PackageError(
                f"{CONFIG_FILE}: span.pii_buckets must reach the span budget of "
                f"{SPAN_TARGET_TOKENS} tokens"
            )
        return cls(
            head_file=span.file("head_file"),
            router_label_sets={
                name: frozenset(sets.texts(name)) for name in sets.value
            },
            schemas={name: _schema(schemas.section(name)) for name in schemas.value},
            default_question=span.text("default_question"),
            threshold=span.number("threshold"),
            threshold_names_only=span.number("threshold_names_only"),
            thresholds_by_set={name: by_set.number(name) for name in by_set.value},
            pii_buckets=buckets,
            evidence=span.optional("evidence", _evidence),
            ner=span.optional("ner", _ner),
            relation=span.optional("relation", _relation),
        )


def _schema(schema: _Section) -> Schema:
    labels = schema.section("labels")
    return {
        "question": schema.text("question"),
        "labels": {name: labels.text(name) for name in labels.value},
    }


def _evidence(evidence: _Section) -> Evidence:
    return Evidence(
        labels=frozenset(x.lower() for x in evidence.texts("labels")),
        question=evidence.text("question"),
        threshold=evidence.number("threshold"),
        decode=evidence.section("decode").value,
    )


def _ner(ner: _Section) -> Ner:
    return Ner(
        head_file=ner.file("head_file"),
        threshold=ner.number("threshold"),
        threshold_names_only=ner.number("threshold_names_only"),
        keeps_names_only=ner.value.get("names_only_route") == "ner",
    )


def _relation(relation: _Section) -> Relation:
    return Relation(
        head_file=relation.file("head_file"),
        threshold_pair=relation.number("threshold_dev_pair"),
        threshold_multi=relation.number("threshold_dev_multi"),
    )


@dataclass(frozen=True)
class _Section:
    value: dict[str, Any]
    where: str

    @classmethod
    def of(cls, value: Any, where: str) -> _Section:
        if not isinstance(value, dict):
            raise PackageError(f"{CONFIG_FILE}: {where} must be an object")
        return cls(value, where)

    def section(self, key: str) -> _Section:
        return _Section.of(self.value.get(key), f"{self.where}.{key}")

    def optional(self, key: str, parse: Callable[[_Section], T]) -> T | None:
        return None if self.value.get(key) is None else parse(self.section(key))

    def sections(self, key: str) -> list[_Section]:
        items = self.value.get(key)
        if not isinstance(items, list):
            self._refuse(key, "a list")
        return [
            _Section.of(item, f"{self.where}.{key}[{index}]")
            for index, item in enumerate(items)
        ]

    def text(self, key: str) -> str:
        value = self.value.get(key)
        if not isinstance(value, str):
            self._refuse(key, "a string")
        return value

    def texts(self, key: str) -> list[str]:
        value = self.value.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            self._refuse(key, "a list of strings")
        return value

    def number(self, key: str) -> float:
        value = self.value.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            self._refuse(key, "a number")
        return float(value)

    def file(self, key: str) -> str:
        return config_file(self.value.get(key), f"{self.where}.{key}")

    def _refuse(self, key: str, kind: str) -> NoReturn:
        raise PackageError(f"{CONFIG_FILE}: {self.where}.{key} must be {kind}")


@dataclass(frozen=True)
class Vela3D3Package:
    """A verified ``vela3/3`` package (its base is verified by the ``decision3`` family)."""

    root: Path
    vela3: dict[str, Any]
    span: SpanConfig
    base: Base
    adapters: dict[str, Adapter]
    router_questions: dict[str, Any]
    files: dict[str, str]
    weights_sha256: str


def _base(vela3: dict[str, Any]) -> Base:
    base = vela3.get("base")
    if not isinstance(base, dict) or not isinstance(base.get("repo_id"), str):
        raise PackageError(f"{CONFIG_FILE}: base must name the Decision 3.0 repository")
    revision = str(base.get("revision"))
    if not REVISION.fullmatch(revision):
        raise PackageError(f"{CONFIG_FILE}: base revision must be a 40-hex commit")
    return Base(base["repo_id"], revision)


def adapter_of(root: Path, name: str, spec: Any) -> Adapter:
    """Read one adapter's header and check it against its ``vela3_config.json`` entry (no tensor is read)."""
    if not isinstance(spec, dict):
        raise PackageError(f"{CONFIG_FILE}: adapters.{name} must be an object")
    file = config_file(spec.get("file"), f"adapters.{name}.file")
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
        match down, up:
            case [down_rank, _], [_, up_rank] if down_rank == up_rank == rank:
                continue
        raise PackageError(
            f"adapter {name}: factor shapes at {module} do not fit rank {rank}"
        )
    return Adapter(name, file, rank, alpha)


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
    span = SpanConfig.parse(vela3.get("span"))
    questions_file = config_file(
        vela3.get("router_questions_file"), "router_questions_file"
    )
    definitions = vela3.get("definitions") or {}
    data = ["config.json", CONFIG_FILE, questions_file]
    if definitions.get("cache_file"):
        data.append(config_file(definitions["cache_file"], "definitions.cache_file"))
    weights = [
        config_file((spec or {}).get("file"), f"adapters.{name}.file")
        for name, spec in specs.items()
        if name not in IMAGE_ADAPTERS
    ]
    weights += list(span.heads.values())
    if definitions.get("writer") == "grafted_head":
        weights.append(
            config_file(definitions.get("lm_head_file"), "definitions.lm_head_file")
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
    if read_json(root / "config.json", mapping=True).get("model_type") != "qwen3_5":
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
        span=span,
        base=base,
        adapters=adapters,
        router_questions=questions,
        files=files,
        weights_sha256=sha256_json({name: files[name] for name in weights}),
    )


def identity(package: Vela3D3Package, base_sha256: str) -> str:
    """The package's identity: its weight files and the base it runs on."""
    return sha256_json({"weights": package.weights_sha256, "base": base_sha256})
