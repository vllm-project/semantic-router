"""Evaluation-row construction shared by the benchmark builders.

A row is the kit row plus ``images`` (``d25.omni.common.vision_format``)::

    {"id", "family", "split", "images": [...], "state", "questions": {"q1": {...}},
     "expected": {"q1": key}, "metadata": {...}, "_evaluation": {"run_id", "group_id", "track"}}

Multiple-choice questions use letter keys ``A, B, ...`` with the option text as the value (``None`` when
the options exist only inside the image); binary questions use two named keys. ``metadata`` carries
``benchmark``, ``subtask``, ``chance``, ``n_options``, ``n_images``, ``tags`` and ``provenance``
(``repo``, ``revision``, ``file``, ``sha256``, ``source_id``). Images are stored as their original
bytes, content-addressed under ``images/`` of the suite root.
"""

from __future__ import annotations

import hashlib
import io
import random
import string
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from d25.omni.common import vision_format
from d25.omni.suite import sources as src

LETTERS = string.ascii_uppercase
FORMATS = {
    "JPEG": "jpg",
    "PNG": "png",
    "WEBP": "webp",
    "GIF": "gif",
    "BMP": "bmp",
    "TIFF": "tif",
    "MPO": "jpg",
}


def stable_rng(*parts: Any) -> random.Random:
    """A ``random.Random`` seeded from the sha256 of the parts (independent of PYTHONHASHSEED)."""
    seed = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(seed[:8], "big"))


def image_info(payload: bytes) -> tuple[str, int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as im:
        fmt = FORMATS.get(im.format or "", (im.format or "bin").lower())
        return fmt, im.width, im.height


class Context:
    """Paths of one build and the content-addressed image store."""

    def __init__(self, sources_root: str | Path, out_root: str | Path):
        self.sources_root = Path(sources_root)
        self.out_root = Path(out_root)
        self.images: dict[str, dict] = {}

    def source(self, key: str) -> Path:
        return self.sources_root / key

    def provenance(self, key: str, file: str, source_id: Any, **extra: Any) -> dict:
        s = src.SOURCES[key]
        out = {
            "repo": s.repo or s.url,
            "revision": s.revision,
            "file": file,
            "sha256": src.file_sha256(key, self.sources_root, file),
            "source_id": str(source_id),
        }
        out.update(extra)
        return out

    def store(self, payload: bytes) -> str:
        """Store image bytes unchanged; returns the path relative to the suite root."""
        digest = vision_format.sha256_bytes(payload)
        if digest not in self.images:
            fmt, width, height = image_info(payload)
            ref = vision_format.store_image(payload, self.out_root, fmt)
            self.images[digest] = {
                "path": ref,
                "width": width,
                "height": height,
                "bytes": len(payload),
            }
        return self.images[digest]["path"]


def letter_criteria(options: Sequence[Any]) -> dict[str, Any]:
    if len(options) > len(LETTERS):
        raise ValueError("too many options")
    return {LETTERS[i]: text for i, text in enumerate(options)}


def make_row(
    *,
    benchmark: str,
    split: str,
    source_id: str,
    images: Sequence[str],
    instructions: str,
    criteria: Mapping[str, Any],
    gold: str,
    provenance: Mapping[str, Any],
    subtask: str | None = None,
    state: Any = None,
    tags: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict:
    if gold not in criteria:
        raise ValueError(
            f"{benchmark}:{source_id}: gold {gold!r} not among {list(criteria)}"
        )
    if len(images) > vision_format.MAX_IMAGES:
        raise ValueError(f"{benchmark}:{source_id}: {len(images)} images")
    rid = ":".join(p for p in (benchmark, split, subtask, source_id) if p)
    metadata = {
        "benchmark": benchmark,
        "subtask": subtask,
        "chance": 1.0 / len(criteria),
        "n_options": len(criteria),
        "n_images": len(images),
        "tags": list(tags),
        "provenance": dict(provenance),
    }
    if extra:
        metadata.update(extra)
    return {
        "id": rid,
        "family": benchmark,
        "split": split,
        "images": list(images),
        "state": state if state is not None else {},
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(criteria),
            }
        },
        "expected": {"q1": gold},
        "metadata": metadata,
        "_evaluation": {"run_id": rid, "group_id": rid, "track": benchmark},
    }


def letter_of(answer: str) -> str:
    """``"(B)"``, ``"B"``, ``"B."`` -> ``"B"``."""
    s = answer.strip().strip("().").strip()
    if len(s) != 1 or s not in LETTERS:
        raise ValueError(f"not a letter answer: {answer!r}")
    return s
