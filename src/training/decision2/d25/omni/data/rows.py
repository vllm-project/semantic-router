"""Training-row construction shared by the Omni corpus generators and converters.

A generator yields ``Item`` objects (images, state, question text, options, gold); ``make_row`` turns one
into a training row of the ``vision_format`` contract with a one-hot gold target, the option order
permuted so that the gold position is uniform, and the images stored content-addressed and re-encoded
at no more than 1,638,400 pixels (PNG for rendered text, JPEG q95 for photos).
"""

from __future__ import annotations

import hashlib
import io
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from d25.omni.common import vision_format

LETTERS = string.ascii_uppercase


def rng_for(*parts: Any) -> random.Random:
    """Deterministic RNG keyed by the item identity (independent of PYTHONHASHSEED)."""
    key = ":".join(str(p) for p in parts).encode("utf-8")
    return random.Random(int(hashlib.sha256(key).hexdigest()[:16], 16))


@dataclass
class Item:
    """One generated question before option permutation and image storage.

    ``options`` lists option texts with the gold first unless ``gold`` says otherwise; ``keys`` gives
    semantic keys (default: letters assigned after permutation); ``fixed_order`` keeps the given order
    (ordinal answers such as left/right pairs or "All other answers are incorrect" last).
    """

    source: str
    family: str
    skill: str
    images: list[Image.Image]
    image_kinds: list[str]
    question: str
    options: list[str] | None = None
    gold: int = 0
    keys: list[str] | None = None
    state: Any = ""
    noul: bool | None = None
    fixed_order: bool = False
    describe_keys: bool = True
    image_text: list[str] = field(default_factory=list)
    orig_sha256: list[str | None] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def encode_image(image: Image.Image, kind: str) -> tuple[bytes, str]:
    """Bytes of ``image`` capped at ``MAX_PIXELS``: ``png`` for rendered text, ``jpg`` for photos."""
    image = image.convert("RGB")
    width, height = image.size
    if width * height > vision_format.MAX_PIXELS:
        scale = (vision_format.MAX_PIXELS / (width * height)) ** 0.5
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        while size[0] * size[1] > vision_format.MAX_PIXELS:
            size = (size[0] - 1, size[1] - 1)
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if kind == "png":
        image.save(buffer, format="PNG", optimize=False, compress_level=6)
        return buffer.getvalue(), "png"
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue(), "jpg"


def store(image: Image.Image, kind: str, root: str | Path) -> tuple[str, str]:
    payload, ext = encode_image(image, kind)
    return vision_format.store_image(payload, root, ext), vision_format.sha256_bytes(
        payload
    )


def make_row(
    item: Item, index: int, root: str | Path, licence: str, source_split: str
) -> dict[str, Any]:
    """Training row for ``item``; images are written under ``root/images``."""
    rng = rng_for(item.source, item.family, index, "order")
    meta: dict[str, Any] = {
        "skill": item.skill,
        "licence": licence,
        "source": item.source,
        "source_split": source_split,
        "verified": True,
        **item.meta,
    }
    if item.noul is not None:
        question = {"type": "noul", "instructions": item.question}
        if item.options:
            question["criteria"] = {"false": item.options[0], "true": item.options[1]}
        target = [0.0, 1.0] if item.noul else [1.0, 0.0]
        label = int(item.noul)
        meta["gold"] = bool(item.noul)
    else:
        options = list(item.options or [])
        if not 2 <= len(options) <= vision_format.text_format.MAX_OPTIONS:
            raise ValueError(f"{item.family}: need 2 to 255 options")
        order = list(range(len(options)))
        if not item.fixed_order:
            gold_position = rng.randrange(len(options))
            others = [i for i in order if i != item.gold]
            rng.shuffle(others)
            order = others[:gold_position] + [item.gold] + others[gold_position:]
        keys = (
            [item.keys[i] for i in order]
            if item.keys
            else list(LETTERS[: len(options)])
        )
        if len(set(keys)) != len(keys):
            raise ValueError(f"{item.family}: duplicate option keys")
        criteria = {
            key: (options[i] if item.describe_keys else None)
            for key, i in zip(keys, order)
        }
        question = {
            "type": "choice",
            "instructions": item.question,
            "criteria": criteria,
        }
        label = order.index(item.gold)
        target = [1.0 if i == label else 0.0 for i in range(len(options))]
        meta["gold"] = keys[label]
        meta["gold_position"] = label
        meta["n_options"] = len(options)
    paths, digests = [], []
    for image, kind in zip(item.images, item.image_kinds):
        path, digest = store(image, kind, root)
        paths.append(path)
        digests.append(digest)
    meta["image_sha256"] = digests
    if any(item.orig_sha256):
        meta["image_orig_sha256"] = list(item.orig_sha256)
    if item.image_text:
        meta["image_text"] = item.image_text
    row = {
        "id": f"{item.source}/{item.family}/{index:07d}",
        "source": item.source,
        "family": item.family,
        "state": item.state if item.state not in (None,) else "",
        "question": question,
        "target": target,
        "label": label,
        "weight": 1.0,
        "images": paths,
        "meta": meta,
    }
    vision_format.validate_row(row)
    return row


def numeric_distractors(
    gold: float,
    rng: random.Random,
    n: int,
    integer: bool | None = None,
    spread: float = 0.35,
) -> list[float]:
    """``n`` distinct plausible numbers near ``gold`` (same sign and precision)."""
    integer = float(gold).is_integer() if integer is None else integer
    out: list[float] = []
    seen = {gold}
    attempts = 0
    while len(out) < n and attempts < 500:
        attempts += 1
        if integer:
            delta = rng.choice([-1, 1]) * max(
                1, int(round(abs(gold) * rng.uniform(0.05, spread)))
            )
            value = float(
                int(gold) + delta
                if rng.random() < 0.7
                else int(gold) + rng.randint(-3, 3)
            )
            if gold >= 0 and value < 0:
                continue
        else:
            value = round(
                gold * (1 + rng.choice([-1, 1]) * rng.uniform(0.05, spread)), 2
            )
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def fmt_number(value: float, decimals: int | None = None) -> str:
    if decimals is not None:
        return f"{value:,.{decimals}f}"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")
