"""Shared image-aware prompt and row contract for Decision 2.5 Omni.

This module extends ``d25.vega.common.decision_format`` instead of forking it: the system prompt,
the 255 single-token answer codes, the option order, the targets and the kit answer conversion are
the Vega ones, so a text-only row renders exactly as it does for Vega.

A vision row adds ``images``: a list of up to ``MAX_IMAGES`` references, each a path relative to the
directory that holds the row file, an absolute path, or a ``data:image/...;base64,`` URL. Images go
before the text in the user turn, one ``{"type": "image"}`` part per image, in list order.

Training row (one question per row): the Vega training row plus ``images`` and, in ``meta``,
``image_sha256`` (one per image), ``licence`` and ``source_split``.

Evaluation row (public vision suite and private proxies): the kit row format plus ``images``::

    {"id": str, "family": str, "split": str, "images": [str, ...],
     "state": str | JSON, "questions": {qid: question}, "expected": {qid: answer},
     "metadata": {...}}
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from d25.vega.common import decision_format as text_format

FORMAT_ID = "d25-omni-code-readout-v1"
MAX_IMAGES = 4
MAX_PIXELS = 1_638_400
MIN_PIXELS = 65_536
PROCESSOR_SIZE = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}

SYSTEM_PROMPT = text_format.SYSTEM_PROMPT
options = text_format.options
answer_codes = text_format.answer_codes
user_prompt = text_format.user_prompt
to_answer = text_format.to_answer


def messages(
    state: Any, question: dict[str, Any], codes: Sequence[str], n_images: int = 0
) -> list[dict[str, Any]]:
    if n_images == 0:
        return text_format.messages(state, question, codes)
    if not 0 < n_images <= MAX_IMAGES:
        raise ValueError(f"a row takes 0 to {MAX_IMAGES} images, got {n_images}")
    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": user_prompt(state, question, codes)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def render(
    processor,
    state: Any,
    question: dict[str, Any],
    codes: Sequence[str],
    n_images: int = 0,
) -> str:
    """Chat-template text with one image placeholder per image (thinking off)."""
    return processor.apply_chat_template(
        messages(state, question, codes, n_images),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def configure_processor(processor, max_pixels: int = MAX_PIXELS):
    """Apply the board image budget (up to 1.6 MP per image) to a Qwen3.5-family processor."""
    processor.image_processor.size = {
        "shortest_edge": MIN_PIXELS,
        "longest_edge": max_pixels,
    }
    return processor


def load_image(ref: Any, root: str | Path | None = None):
    from PIL import Image

    if isinstance(ref, Image.Image):
        return ref.convert("RGB")
    if isinstance(ref, str) and ref.startswith("data:image/"):
        payload = base64.b64decode(ref.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    path = Path(ref)
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    with Image.open(path) as image:
        return image.convert("RGB")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def store_image(payload: bytes, root: str | Path, ext: str) -> str:
    """Content-addressed copy under ``root/images``; returns the path relative to ``root``."""
    digest = sha256_bytes(payload)
    relative = Path("images") / digest[:2] / f"{digest}.{ext.lstrip('.').lower()}"
    target = Path(root) / relative
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(target)
    return relative.as_posix()


def requests(row: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Per-question engine inputs ``{state, question, images}`` of an evaluation row."""
    images = list(row.get("images") or [])
    for qid, question in row["questions"].items():
        yield qid, {"state": row.get("state"), "question": question, "images": images}


def validate_row(row: dict[str, Any]) -> None:
    """Training-row check: the Vega contract plus the image list."""
    text_format.validate_row(row)
    images = row.get("images") or []
    if len(images) > MAX_IMAGES or any(
        not isinstance(ref, str) or not ref for ref in images
    ):
        raise ValueError(
            f"row {row['id']}: images must be 0 to {MAX_IMAGES} non-empty strings"
        )
    digests = (row.get("meta") or {}).get("image_sha256")
    if images and (not isinstance(digests, list) or len(digests) != len(images)):
        raise ValueError(
            f"row {row['id']}: meta.image_sha256 needs one digest per image"
        )
