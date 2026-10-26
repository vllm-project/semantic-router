"""System One ``assets``: images a request names by reference and places anywhere text can go.

A request's top-level ``assets`` maps each reference string to an image
(``data_base64``, or a multipart part the wire layer has already attached as
bytes). Every reference is replaced wherever it occurs in the strings of the
state, of a question's ``instructions`` and of its options (criteria values,
``choices`` / ``labels`` descriptions, ``levels``), in the request and in
every ``states`` entry; a string equal to a reference is that image, and a
longer one is split at each non-overlapping occurrence, the longest
reference winning at one position. Object keys are never replaced. Each
occurrence is one use: it adds the image again where it sits. Every asset
must be used.

``prepare`` checks the assets and these rules for the whole request, before
its states are split, and replaces ``assets`` with the checked ``Asset``
objects. ``resolve`` then turns one state's strings into ``Segments`` for its
model, and ``hoist`` gives a model that reads images only in front of the
state (today's families) the state's uses as images and a label in the text.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import MediaError, RequestError

MAX_ASSET_BYTES = 16 << 20
MAX_REFERENCE_BYTES = 1024
MAX_REFERENCES_BYTES = 65536
ASSET_FIELDS = frozenset({"type", "format", "data_base64", "part", "data"})
FORMATS = ("png", "jpg", "jpeg")
# Option fields whose values (or descriptions) may hold references.
OPTION_FIELDS = ("criteria", "choices", "levels", "labels")
PLACEMENTS = ("state", "instructions", "criteria")
# File signatures and the header sizes the sniffer reads.
PNG_HEADER = 33
JPEG_MARKER = 0xFF
JPEG_RESTART = range(0xD0, 0xD8)
JPEG_FRAMES = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
WEBP_VP8_HEADER = 30
WEBP_VP8L_HEADER = 25


@dataclass(frozen=True, eq=False)
class Asset:
    """One checked image of a request: its reference, bytes, format (from the bytes) and SHA-256."""

    reference: str
    data: bytes = field(repr=False)
    format: str
    sha256: str
    width: int
    height: int
    # The base64 text the request sent, kept so a data URL needs no second encoding.
    text: str | None = field(default=None, repr=False)

    def data_url(self) -> str:
        text = (
            self.text
            if self.text is not None
            else base64.b64encode(self.data).decode("ascii")
        )
        return f"data:image/{self.format};base64,{text}"


@dataclass(frozen=True)
class Use:
    """One occurrence of a reference: the asset, where it sits and, for a question's, the question."""

    asset: Asset
    placement: str
    question: str | None = None


@dataclass(frozen=True)
class Segments:
    """A string split at its references: text and ``Use`` parts in order (no two text parts adjacent)."""

    parts: tuple[str | Use, ...]

    def uses(self) -> list[Use]:
        return [part for part in self.parts if isinstance(part, Use)]


@dataclass
class Resolved:
    """One state's request with every referenced string as ``Segments``; the uses in reading order."""

    state: Any
    questions: dict[str, Any]
    uses: list[Use]

    def state_uses(self) -> list[Use]:
        return [use for use in self.uses if use.placement == "state"]

    def question_uses(self, question: str) -> list[Use]:
        return [use for use in self.uses if use.question == question]


# ------------------------------------------------------------------ checking


def prepare(body: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """``body`` with its ``assets`` checked for the whole request (``states`` included) and replaced by ``Asset`` objects."""
    entries = body.get("assets")
    if entries is None:
        return body
    if not isinstance(entries, dict) or not entries:
        raise RequestError(
            "asset_malformed",
            "assets must be a nonempty object of references to images",
        )
    total = 0
    assets: dict[str, Asset] = {}
    for reference, entry in entries.items():
        size = len(reference.encode("utf-8"))
        if not reference.strip() or size > MAX_REFERENCE_BYTES:
            raise RequestError(
                "asset_reference",
                f"asset references are 1 to {MAX_REFERENCE_BYTES:,} bytes and not blank",
                MAX_REFERENCE_BYTES,
            )
        total += size
        assets[reference] = _asset(reference, entry)
    if total > MAX_REFERENCES_BYTES:
        raise RequestError(
            "asset_reference",
            f"asset references take {total:,} bytes together, over {MAX_REFERENCES_BYTES:,}",
            MAX_REFERENCES_BYTES,
        )
    finder = Finder(assets)
    used: dict[str, int] = dict.fromkeys(assets, 0)
    expanded = 0
    for part in _parts(body):
        for use in finder.uses_in(part.get("state"), part.get("questions")):
            used[use.asset.reference] += 1
            expanded += len(use.asset.data)
    unused = [reference for reference, count in used.items() if not count]
    if unused:
        raise RequestError(
            "asset_unreferenced",
            f"assets[{unused[0][:64]!r}] is not referenced in the state, instructions or criteria; "
            "every asset must be used",
        )
    if expanded > max_bytes:
        raise RequestError(
            "asset_uses",
            f"the request's image uses take {expanded:,} bytes, over {max_bytes:,}; every use of "
            "an image counts again, so put an image used by several questions in the state",
            max_bytes,
        )
    return {**body, "assets": assets}


def _parts(body: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield body
    states = body.get("states")
    if isinstance(states, dict):
        for entry in states.values():
            if isinstance(entry, dict):
                yield entry


def _asset(reference: str, entry: Any) -> Asset:
    where = f"assets[{reference[:64]!r}]"
    if not isinstance(entry, dict) or set(entry) - ASSET_FIELDS:
        raise RequestError(
            "asset_malformed",
            f"{where} takes type, format and data_base64 (or a multipart part)",
        )
    if entry.get("type", "image") != "image":
        raise RequestError("asset_malformed", f"{where}: type must be image")
    if entry.get("format") is not None and entry["format"] not in FORMATS:
        raise RequestError(
            "asset_malformed", f"{where}: format must be png, jpg or jpeg"
        )
    text = None
    data = entry.get("data")
    if data is not None and not isinstance(data, bytes):
        raise RequestError("asset_malformed", f"{where}: unknown field data")
    if data is None:
        if "part" in entry:
            raise RequestError(
                "multipart_malformed",
                f"{where} names a part, which needs a multipart/form-data body",
            )
        text = entry.get("data_base64")
        if not isinstance(text, str) or not text:
            raise RequestError(
                "asset_malformed", f"{where} needs data_base64 or a multipart part"
            )
        try:
            data = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise MediaError(
                "image_malformed", f"{where}: data_base64 is not base64"
            ) from exc
    if len(data) > MAX_ASSET_BYTES:
        raise MediaError(
            "image_too_large",
            f"{where} has {len(data):,} bytes, over {MAX_ASSET_BYTES:,}",
            MAX_ASSET_BYTES,
        )
    kind, width, height = sniff(data, where)
    return Asset(
        reference, data, kind, hashlib.sha256(data).hexdigest(), width, height, text
    )


def sniff(data: bytes, where: str) -> tuple[str, int, int]:
    """The format (from the bytes) and size of an image file, without decoding it; animated images are refused."""
    if (
        data.startswith(b"\x89PNG\r\n\x1a\n")
        and len(data) >= PNG_HEADER
        and data[12:16] == b"IHDR"
    ):
        if _png_animated(data):
            raise MediaError(
                "image_format", f"{where} is an animated PNG; send one frame"
            )
        return (
            "png",
            int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"),
        )
    if data.startswith(b"\xff\xd8\xff"):
        width, height = _jpeg_size(data, where)
        return "jpeg", width, height
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        width, height, animated = _webp_size(data, where)
        if animated:
            raise MediaError(
                "image_format", f"{where} is an animated WebP; send one frame"
            )
        return "webp", width, height
    raise MediaError(
        "image_format",
        f"{where} is not a PNG, JPEG or WebP image (detected from its bytes)",
    )


def _png_animated(data: bytes) -> bool:
    position = 8
    while position + 8 <= len(data):
        length = int.from_bytes(data[position : position + 4], "big")
        kind = data[position + 4 : position + 8]
        if kind == b"acTL":
            return True
        if kind in (b"IDAT", b"IEND"):
            return False
        position += 12 + length
    return False


def _jpeg_size(data: bytes, where: str) -> tuple[int, int]:
    position = 2
    while position + 4 <= len(data):
        if data[position] != JPEG_MARKER:
            position += 1
            continue
        marker = data[position + 1]
        if marker in (0xD8, 0x01) or marker in JPEG_RESTART:
            position += 2
            continue
        length = int.from_bytes(data[position + 2 : position + 4], "big")
        if marker in JPEG_FRAMES:
            if position + 9 > len(data):
                break
            height = int.from_bytes(data[position + 5 : position + 7], "big")
            width = int.from_bytes(data[position + 7 : position + 9], "big")
            return width, height
        position += 2 + length
    raise MediaError("image_malformed", f"{where} is a JPEG without a frame header")


def _webp_size(data: bytes, where: str) -> tuple[int, int, bool]:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= WEBP_VP8_HEADER:
        animated = bool(data[20] & 0x02)
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height, animated
    if chunk == b"VP8 " and len(data) >= WEBP_VP8_HEADER:
        return (
            int.from_bytes(data[26:28], "little") & 0x3FFF,
            int.from_bytes(data[28:30], "little") & 0x3FFF,
            False,
        )
    if chunk == b"VP8L" and len(data) >= WEBP_VP8L_HEADER:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, False
    raise MediaError("image_malformed", f"{where} is a WebP without a size header")


# ------------------------------------------------------------------ references


class Finder:
    """Finds a request's references in its strings: whole values first, then the longest match left to right."""

    def __init__(self, assets: Mapping[str, Asset]):
        self.assets = assets
        ordered = sorted(assets, key=len, reverse=True)
        self.pattern = re.compile(
            "|".join(re.escape(reference) for reference in ordered)
        )

    def split(self, text: str, placement: str, question: str | None) -> str | Segments:
        asset = self.assets.get(text)
        if asset is not None:
            return Segments((Use(asset, placement, question),))
        parts: list[str | Use] = []
        last = 0
        for match in self.pattern.finditer(text):
            if match.start() > last or not parts:
                parts.append(text[last : match.start()])
            parts.append(Use(self.assets[match.group(0)], placement, question))
            last = match.end()
        if not parts:
            return text
        if last < len(text):
            parts.append(text[last:])
        return Segments(tuple(part for part in parts if part != ""))

    def value(self, value: Any, placement: str, question: str | None = None) -> Any:
        """``value`` with every referenced string as ``Segments`` (object keys untouched)."""
        if isinstance(value, str):
            return self.split(value, placement, question)
        if isinstance(value, list):
            return [self.value(item, placement, question) for item in value]
        if isinstance(value, dict):
            return {
                key: self.value(item, placement, question)
                for key, item in value.items()
            }
        return value

    def question(self, question_id: str, question: Any) -> Any:
        if not isinstance(question, dict):
            return question
        resolved = dict(question)
        if "instructions" in question:
            resolved["instructions"] = self.value(
                question["instructions"], "instructions", question_id
            )
        for name in OPTION_FIELDS:
            if name not in question:
                continue
            options = question[name]
            if name in ("choices", "labels") and isinstance(options, list):
                resolved[name] = [
                    (
                        {
                            **option,
                            "description": self.value(
                                option["description"], "criteria", question_id
                            ),
                        }
                        if isinstance(option, dict) and "description" in option
                        else option
                    )
                    for option in options
                ]
            else:
                resolved[name] = self.value(options, "criteria", question_id)
        return resolved

    def uses_in(self, state: Any, questions: Any) -> list[Use]:
        uses = _uses(self.value(state, "state"))
        if isinstance(questions, dict):
            for question_id, question in questions.items():
                uses += _uses(self.question(question_id, question))
        return uses


def _uses(value: Any) -> list[Use]:
    if isinstance(value, Segments):
        return value.uses()
    if isinstance(value, list):
        return [use for item in value for use in _uses(item)]
    if isinstance(value, dict):
        return [use for item in value.values() for use in _uses(item)]
    return []


def resolve(
    state: Any, questions: dict[str, Any], assets: Mapping[str, Asset]
) -> Resolved:
    """One state's request with its references as ``Segments``, and its uses in reading order (none without assets)."""
    if not assets:
        return Resolved(state, dict(questions), [])
    finder = Finder(assets)
    resolved_state = finder.value(state, "state")
    resolved_questions = {
        qid: finder.question(qid, question) for qid, question in questions.items()
    }
    uses = _uses(resolved_state)
    for question in resolved_questions.values():
        uses += _uses(question)
    return Resolved(resolved_state, resolved_questions, uses)


# ------------------------------------------------------------------ the default placement

# The text that stands for a hoisted image in the state: "[image 2]" is the second image in front of it.
LABEL = "[image {}]"


def hoist(state: Any, first: int) -> tuple[Any, list[Asset]]:
    """The state with each use as its label and the used assets in reading order; labels count from ``first + 1``."""
    hoisted: list[Asset] = []

    def text(value: Any) -> Any:
        if isinstance(value, Segments):
            out = []
            for part in value.parts:
                if isinstance(part, Use):
                    hoisted.append(part.asset)
                    out.append(LABEL.format(first + len(hoisted)))
                else:
                    out.append(part)
            return "".join(out)
        if isinstance(value, list):
            return [text(item) for item in value]
        if isinstance(value, dict):
            return {key: text(item) for key, item in value.items()}
        return value

    return text(state), hoisted


def plain(value: Any) -> Any:
    """``value`` with every ``Segments`` back to the request's own text (references kept)."""
    if isinstance(value, Segments):
        return "".join(
            part if isinstance(part, str) else part.asset.reference
            for part in value.parts
        )
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


# ------------------------------------------------------------------ Box targets


@dataclass(frozen=True)
class Target:
    """What a Box or track question is about: an image (index in its question's view) or the state's video."""

    kind: str  # "image" or "video"
    index: int


def box_target(
    question: dict[str, Any], question_uses: list[Use], state_images: int, videos: int
) -> Target | str:
    """The target of a valid box / track question, or the reason it has none (contract 4.1).

    A question's view is the state's images (``images`` then state uses) followed by its own uses.
    """
    if question.get("type") == "track" or question.get("at") is not None:
        if videos == 1:
            return Target("video", 0)
        return "video_needed" if videos == 0 else "video_target_ambiguous"
    own = [use for use in question_uses if use.placement == "instructions"]
    if len(own) == 1:
        return Target("image", state_images + question_uses.index(own[0]))
    if own:
        return "box_target_ambiguous"
    if state_images == 1:
        return Target("image", 0)
    return "box_needs_image" if state_images == 0 else "box_target_ambiguous"


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
