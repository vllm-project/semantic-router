"""The Vela 3.0 encoder package (vela3-fast): what it holds and how its bytes are verified.

``vela3_config.json`` names the format (``vela3-encoder/1``) and the
adapters: span adapters (``adapters``, ``span.evidence.adapter``,
``span.task_adapters``, ``span.set_adapter``) and router adapters
(``adapters.router``, ``router.task_adapters``), each a safetensors file of
LoRA factors, the top layers it trained and its readout over the frozen
UltraBERT backbone in ``model.safetensors``. An adapter whose file the package
does not ship routes off, as the released runtime does. The family loads the
two configs, the router questions, the tokenizer and the weights; every loaded
byte must match the built-in table's digest for a pinned revision and the
package's ``SHA256SUMS``; bundled Python is never read. The identity is the
SHA-256 of the weight files' digests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import named_files, read_json, safe_relative, sha256_json

ENCODER = "vela3-encoder/1"
CONFIG_FILE = "vela3_config.json"
SUMS_FILE = "SHA256SUMS"
BACKBONE_FILE = "model.safetensors"
COMMON_FILES = ("config.json", CONFIG_FILE, "router_questions.json", "tokenizer.json")
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def member_of(root: Path) -> str | None:
    """The format a package directory holds, read from its ``vela3_config.json`` only."""
    try:
        config = json.loads((Path(root) / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    return ENCODER if config.get("format") == ENCODER else None


@dataclass(frozen=True)
class Adapter:
    """One adapter network: its file, LoRA rank and alpha (from the file's metadata)."""

    name: str
    file: str
    rank: int
    alpha: float


@dataclass(frozen=True)
class Roster:
    """Which adapter answers what (``Vela3Encoder``'s construction).

    ``span``, ``evidence`` and ``span_set`` name span adapters (``evidence``
    is ``span`` itself when the package ships no evidence adapter, and
    ``span_set`` None without a set adapter); ``tasks`` the span task
    adapters (relation, ner); ``router`` the shared router adapter and
    ``router_tasks`` each router question ID's own adapters (several: their
    logits are averaged).
    """

    span: str
    evidence: str
    span_set: str | None
    tasks: dict[str, str]
    router: str
    router_tasks: dict[str, tuple[str, tuple[str, ...]]]


def _file(value: Any, where: str) -> str:
    if not isinstance(value, str) or not safe_relative(value):
        raise PackageError(f"{CONFIG_FILE}: {where} must name a file in the package")
    return value


def roster(root: Path, config: dict[str, Any]) -> tuple[Roster, dict[str, str]]:
    """The routing roster and every adapter's file, by adapter name; files the package lacks route off."""
    present = {name for name in _named(config) if (Path(root) / name).is_file()}
    span_config = config.get("span") or {}
    adapters = config.get("adapters") or {}
    files: dict[str, str] = {}

    def take(name: str, value: Any, where: str) -> str | None:
        file = _file(value, where)
        if file not in present:
            return None
        files[name] = file
        return name

    span = take("span", (adapters.get("span") or {}).get("file"), "adapters.span.file")
    router = take(
        "router", (adapters.get("router") or {}).get("file"), "adapters.router.file"
    )
    if span is None or router is None:
        raise PackageError("the package must ship its span and router adapters")
    evidence = span
    evidence_file = (span_config.get("evidence") or {}).get("adapter")
    if evidence_file:
        evidence = take("evidence", evidence_file, "span.evidence.adapter") or span
    tasks = {}
    for task, file in (span_config.get("task_adapters") or {}).items():
        if take(task, file, f"span.task_adapters.{task}"):
            tasks[task] = task
    span_set = None
    set_config = span_config.get("set_adapter")
    if set_config:
        span_set = take("span_set", set_config.get("file"), "span.set_adapter.file")
    router_tasks: dict[str, tuple[str, tuple[str, ...]]] = {}
    for task, entry in (
        (config.get("router") or {}).get("task_adapters") or {}
    ).items():
        listed = entry.get("files") or [entry.get("file")]
        names = [
            f"router:{task}" if len(listed) == 1 else f"router:{task}:{index}"
            for index in range(len(listed))
        ]
        if not all(_file(f, f"router.task_adapters.{task}") in present for f in listed):
            continue
        for name, file in zip(names, listed, strict=True):
            files[name] = file
        for question_id in entry.get("ids") or ():
            router_tasks[str(question_id)] = (task, tuple(names))
    return Roster(span, evidence, span_set, tasks, router, router_tasks), files


def _named(config: dict[str, Any]) -> list[str]:
    """Every file name the config mentions as an adapter (present or not)."""
    names = [
        entry.get("file")
        for entry in (config.get("adapters") or {}).values()
        if isinstance(entry, dict)
    ]
    span_config = config.get("span") or {}
    names.append((span_config.get("evidence") or {}).get("adapter"))
    names.extend((span_config.get("task_adapters") or {}).values())
    names.append((span_config.get("set_adapter") or {}).get("file"))
    for entry in ((config.get("router") or {}).get("task_adapters") or {}).values():
        names.extend(entry.get("files") or [entry.get("file")])
    return [name for name in names if isinstance(name, str) and safe_relative(name)]


def adapter_metadata(path: Path) -> tuple[int, float]:
    """An adapter file's LoRA rank and alpha, from its safetensors header (no tensor is read)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    try:
        return int(metadata["lora_rank"]), float(metadata["lora_alpha"])
    except (KeyError, ValueError) as exc:
        raise PackageError(f"{path.name} names no LoRA rank and alpha") from exc


@dataclass(frozen=True)
class Vela3Package:
    """A verified Vela 3.0 encoder package."""

    root: Path
    member: str
    config: dict[str, Any]
    vela3: dict[str, Any]
    router_questions: dict[str, Any]
    roster: Roster
    adapters: dict[str, Adapter]
    files: dict[str, str]
    model_sha256: str

    @property
    def max_input_tokens(self) -> int:
        return int(self.config["decision"]["max_len"])

    @property
    def weights(self) -> tuple[Path, ...]:
        return (self.root / BACKBONE_FILE,)


def _sums(root: Path) -> dict[str, str]:
    path = root / SUMS_FILE
    if not path.is_file():
        return {}
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition(" ")
        name = name.strip().lstrip("*")
        if digest and name:
            sums[name] = digest
    return sums


def verify(root: Path, expected: dict[str, str] | None) -> Vela3Package:
    """Check every loaded file; ``expected`` is the built-in table's digests for a pinned revision."""
    root = Path(root)
    member = member_of(root)
    if member is None:
        raise PackageError(
            f"{CONFIG_FILE} does not describe a Vela 3.0 encoder package"
        )
    vela3 = read_json(root / CONFIG_FILE, mapping=True)
    team, adapter_files = roster(root, vela3)
    weight_names = [BACKBONE_FILE, *sorted(set(adapter_files.values()))]
    files = named_files(root, [*COMMON_FILES, *weight_names])
    if expected is not None:
        wrong = sorted(
            name
            for name in set(expected) | set(files)
            if expected.get(name) != files.get(name)
        )
        if wrong:
            raise PackageError(f"files differ from the pinned revision: {wrong[:5]}")
    sums = _sums(root)
    mismatched = sorted(
        name for name, digest in files.items() if name in sums and sums[name] != digest
    )
    if mismatched:
        raise PackageError(f"files differ from {SUMS_FILE}: {mismatched[:5]}")
    config = read_json(root / "config.json", mapping=True)
    if config.get("model_type") != "ultrabert" or "decision" not in config:
        raise PackageError(
            "config.json does not describe the UltraBERT decision backbone"
        )
    questions = read_json(root / "router_questions.json", mapping=True).get("questions")
    if not isinstance(questions, dict):
        raise PackageError("router_questions.json lists no router questions")
    adapters = {}
    for name, file in adapter_files.items():
        rank, alpha = adapter_metadata(root / file)
        adapters[name] = Adapter(name, file, rank, alpha)
    return Vela3Package(
        root=root,
        member=member,
        config=config,
        vela3=vela3,
        router_questions=questions,
        roster=team,
        adapters=adapters,
        files=files,
        model_sha256=sha256_json({name: files[name] for name in weight_names}),
    )
