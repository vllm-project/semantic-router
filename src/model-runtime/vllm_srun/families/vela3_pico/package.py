"""The Vela 3.1 pico package (``vela3-pico/1``): what it holds and how its bytes are verified.

A pico is one UltraBERT encoder fine-tuned whole, with one readout over it
(option and word projections) in ``model.safetensors``; it ships no adapters.
``vela3_config.json`` names the format and the span rules, ``config.json`` the
backbone and the eight markers, ``router_questions.json`` the built-in router
questions. The family loads those, the tokenizer and the weights; every loaded
byte must match the built-in table's digests for a pinned revision, and an
unpinned package must ship a ``SHA256SUMS`` that covers them (a pinned
download fetches only the table's files, so it needs none). Bundled Python
(``render31.py``, ``vela3_encoder.py``, ``pico_schema.py``) and the int8 ONNX
encoder are never read. The identity is the SHA-256 of the weight file's
digest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import (
    SUMS_FILE,
    named_files,
    read_json,
    read_sums,
    sha256_json,
)
from ..vela3_shared import CONFIG_FILE, check_markers

FORMAT = "vela3-pico/1"
WEIGHTS_FILE = "model.safetensors"
LOADED_FILES = (
    "config.json",
    CONFIG_FILE,
    "router_questions.json",
    "tokenizer.json",
    WEIGHTS_FILE,
)


def member_of(root: Path) -> str | None:
    """The format a package directory holds, read from its ``vela3_config.json`` only."""
    try:
        config = json.loads((Path(root) / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    return FORMAT if config.get("format") == FORMAT else None


@dataclass(frozen=True)
class PicoPackage:
    """A verified pico package."""

    root: Path
    config: dict[str, Any]
    pico: dict[str, Any]
    router_questions: dict[str, Any]
    files: dict[str, str]
    model_sha256: str

    @property
    def max_input_tokens(self) -> int:
        return int(self.config["decision"]["max_len"])


def verify(root: Path, expected: dict[str, str] | None) -> PicoPackage:
    """Check every loaded file; ``expected`` is the built-in table's digests for a pinned revision."""
    root = Path(root)
    if member_of(root) is None:
        raise PackageError(f"{CONFIG_FILE} does not describe a {FORMAT} package")
    files = named_files(root, list(LOADED_FILES))
    if expected is not None:
        wrong = sorted(
            name
            for name in set(expected) | set(files)
            if expected.get(name) != files.get(name)
        )
        if wrong:
            raise PackageError(f"files differ from the pinned revision: {wrong[:5]}")
    sums = read_sums(root, required=expected is None)
    if expected is None:
        unlisted = sorted(name for name in files if name not in sums)
        if unlisted:
            raise PackageError(f"{SUMS_FILE} does not list {unlisted[:5]}")
    mismatched = sorted(
        name for name, digest in files.items() if name in sums and sums[name] != digest
    )
    if mismatched:
        raise PackageError(f"files differ from {SUMS_FILE}: {mismatched[:5]}")
    config = read_json(root / "config.json", mapping=True)
    if config.get("model_type") != "ultrabert" or not isinstance(
        config.get("decision"), dict
    ):
        raise PackageError(
            "config.json does not describe the UltraBERT decision backbone"
        )
    check_markers(config)
    pico = read_json(root / CONFIG_FILE, mapping=True)
    if not isinstance(pico.get("span"), dict) or not isinstance(
        pico.get("router"), dict
    ):
        raise PackageError(f"{CONFIG_FILE} has no span and router rules")
    questions = read_json(root / "router_questions.json", mapping=True).get("questions")
    if not isinstance(questions, dict):
        raise PackageError("router_questions.json lists no router questions")
    return PicoPackage(
        root=root,
        config=config,
        pico=pico,
        router_questions=questions,
        files=files,
        model_sha256=sha256_json({WEIGHTS_FILE: files[WEIGHTS_FILE]}),
    )
