"""The ``/v1/embeddings`` surface shared by every family that serves it.

Request parsing (OpenAI-compatible ``input`` with text, image and audio
content parts, ``dimensions``, ``layer``, ``input_type``,
``encoding_format``), token budgets inside the tokenizer's special-token
envelope, pooling and Matryoshka views of hidden states, content-hash cache
keys, and response assembly with the representation identity that keeps
vector spaces apart.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from ..errors import (
    DEADLINE_EXCEEDED,
    INVALID_INPUT,
    INVALID_MODEL_OUTPUT,
    MAX_LENGTH_EXCEEDED,
)
from ..plugins.base import DEADLINE, EmbeddingInfo, SurfacePlan, SurfaceRequest
from ..text import bounds
from .task import positive_option

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_INPUTS = 2048
MAX_MEDIA_BYTES = 16 << 20
NORM_EPSILON = 1e-12
ENCODINGS = ("float", "base64")
OVERFLOWS = ("reject", "truncate")
_MEDIA_TYPES = {"image_url": "image", "input_audio": "audio"}


@dataclass(frozen=True)
class EmbeddingInput:
    """One input: its text, or its decoded media bytes; ``error`` when it cannot be read."""

    index: int
    modality: str
    text: str | None = None
    data: bytes | None = None
    media_type: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class EmbeddingRequest:
    """A parsed embeddings request; ``dimension`` and ``layer`` are ones the model declares.

    ``layer`` is 0 for a model without layer exits (its one published readout).
    """

    inputs: tuple[EmbeddingInput, ...]
    dimension: int
    layer: int
    input_type: str | None
    encoding: str
    overflow: str
    max_tokens: int


@dataclass(frozen=True)
class EmbedItem:
    """One media or text input ready to run.

    ``ids`` bound admission and batches; media have none and set ``cost``, the
    text tokens of the same model their forward takes as long as.
    """

    index: int
    modality: str
    ids: list[int]
    cache_key: str
    features: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    cost: int | None = None


@dataclass
class EmbeddingPlanState:
    """What ``finish`` needs: the request and, per input, its item position or error, and its usage."""

    request: EmbeddingRequest
    slots: list[int | str]
    usages: list[dict[str, Any] | None]
    representation: dict[str, Any]


def _media(index: int, part: dict[str, Any]) -> EmbeddingInput:
    kind = cast(str, part.get("type"))
    modality = _MEDIA_TYPES[kind]
    payload = part.get(kind)
    if not isinstance(payload, dict):
        raise ValueError(f"input[{index}].{kind} must be an object")
    if modality == "image":
        url = payload.get("url")
        if not isinstance(url, str) or not url.startswith("data:"):
            raise ValueError(f"input[{index}].image_url.url must be a data URL")
        header, separator, encoded = url.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ValueError(f"input[{index}].image_url.url must be base64 data")
        media_type = header[len("data:") : -len(";base64")].lower()
    else:
        audio, audio_format = payload.get("data"), payload.get("format", "wav")
        if not isinstance(audio, str) or not isinstance(audio_format, str):
            raise ValueError(
                f"input[{index}].input_audio needs data and format strings"
            )
        encoded, media_type = audio, audio_format.lower()
    if len(encoded) > (MAX_MEDIA_BYTES * 4) // 3 + 4:
        return EmbeddingInput(index, modality, error=INVALID_INPUT)
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return EmbeddingInput(
            index, modality, media_type=media_type, error=INVALID_INPUT
        )
    return EmbeddingInput(index, modality, data=data, media_type=media_type)


def _inputs(value: Any, modalities: Sequence[str]) -> list[EmbeddingInput]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError("input must be a string or a nonempty list")
    if len(value) > MAX_INPUTS:
        raise ValueError(f"input holds at most {MAX_INPUTS} items")
    inputs = []
    for index, part in enumerate(value):
        kind = part.get("type") if isinstance(part, dict) else None
        if not isinstance(part, str) and kind not in ("text", *_MEDIA_TYPES):
            raise ValueError(
                f"input[{index}] must be a string or a text, image_url or input_audio part"
            )
        modality = "text" if kind is None else _MEDIA_TYPES.get(kind, "text")
        if modality not in modalities:
            raise ValueError(f"this model does not embed {modality} input")
        if isinstance(part, str):
            inputs.append(EmbeddingInput(index, "text", text=part))
        elif kind == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(f"input[{index}].text must be a string")
            inputs.append(EmbeddingInput(index, "text", text=text))
        else:
            inputs.append(_media(index, part))
    return inputs


def parse_request(
    request: SurfaceRequest, info: EmbeddingInfo, max_input_tokens: int
) -> EmbeddingRequest:
    """Validate an embeddings request against the model's descriptor (``ValueError`` -> 400)."""
    body, options = request.body, request.options
    dimension = (
        positive_option(body.get("dimensions"), "dimensions") or info.dimensions[0]
    )
    if dimension not in info.dimensions:
        raise ValueError(f"dimensions must be one of {list(info.dimensions)}")
    layer = positive_option(body.get("layer"), "layer")
    if not info.layers:
        if layer is not None:
            raise ValueError("this model has no layer exits")
        layer = 0
    elif layer is None:
        layer = info.layers[-1]
    elif layer not in info.layers:
        raise ValueError(f"layer must be one of {list(info.layers)}")
    input_type = body.get("input_type")
    if input_type is not None and input_type not in info.input_types:
        raise ValueError(
            f"input_type must be one of {list(info.input_types)}"
            if info.input_types
            else "this model takes no input_type"
        )
    encoding = body.get("encoding_format", "float")
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding_format must be one of {list(ENCODINGS)}")
    overflow = options.get("overflow", "reject")
    if overflow not in OVERFLOWS:
        raise ValueError(f"options.overflow must be one of {list(OVERFLOWS)}")
    max_tokens = positive_option(options.get("max_tokens"), "options.max_tokens")
    if max_tokens is not None and max_tokens > max_input_tokens:
        raise ValueError(f"options.max_tokens must be at most {max_input_tokens}")
    return EmbeddingRequest(
        inputs=tuple(_inputs(body.get("input"), info.modalities)),
        dimension=dimension,
        layer=layer,
        input_type=input_type,
        encoding=encoding,
        overflow=overflow,
        max_tokens=max_tokens or max_input_tokens,
    )


def encode_text(
    backend: Any, text: str, budget: int, overflow: str
) -> tuple[list[int], dict[str, Any]] | str:
    """Token IDs with the tokenizer's special tokens, within ``budget``; an item error code otherwise.

    ``truncate`` keeps the beginning of the content inside the special-token
    envelope; nothing is cut silently: the usage reports both counts. Only the
    tokens that decide the budget are read (``bounds.read``); for a longer
    text, ``tokens`` counts those and ``tokens_lower_bound`` says so, and a
    text certainly over the budget fails unread unless it is truncated.
    """
    specials = backend.num_special_tokens_to_add(False)
    if overflow != "truncate" and bounds.surely_over(backend, text, budget - specials):
        return MAX_LENGTH_EXCEEDED
    read = bounds.read(backend, text, budget - specials + 1)
    content = read.encoding
    tokens = read.tokens + specials
    if tokens > budget:
        if overflow != "truncate" or budget <= specials:
            return MAX_LENGTH_EXCEEDED
        content.truncate(budget - specials)
    ids = list(backend.post_process(content).ids)
    usage: dict[str, Any] = {
        "tokens": tokens,
        "processed_tokens": len(ids),
        "truncated": len(ids) < tokens,
    }
    if not read.complete:
        usage["tokens_lower_bound"] = True
    return ids, usage


def content_key(*parts: Any) -> str:
    """SHA-256 over typed parts (strings, integers, byte strings, integer sequences)."""
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            raw = part
        elif isinstance(part, str):
            raw = part.encode()
        elif isinstance(part, int):
            raw = str(part).encode()
        else:
            raw = np.asarray(part, dtype="<i8").tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def pool(hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Sentence vectors [batch, hidden] in FP32 from right-padded hidden states."""
    if pooling == "cls":
        return hidden[:, 0].float()
    if pooling == "last_token":
        last = mask.sum(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
    if pooling != "mean":
        raise ValueError(f"unknown pooling {pooling!r}")
    weights = mask.to(torch.float32)
    summed = (hidden.to(torch.float32) * weights.unsqueeze(-1)).sum(1)
    return summed / weights.sum(1, keepdim=True)


def matryoshka(
    vectors: torch.Tensor, dimension: int, normalize: bool = True
) -> torch.Tensor:
    """The first ``dimension`` channels, L2-normalized after truncation (``norm + 1e-12``)."""
    view = vectors[:, :dimension]
    if not normalize:
        return view
    normalized: torch.Tensor = view / (view.norm(dim=-1, keepdim=True) + NORM_EPSILON)
    return normalized


def encode_vector(vector: Sequence[float], encoding: str) -> list[float] | str:
    if encoding == "base64":
        return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode()
    return list(vector)


def representation(
    model_sha256: str,
    layer: int,
    dimension: int,
    normalized: bool,
    modality: str | None = None,
) -> dict[str, Any]:
    """The identity of an embedding space: vectors with different representations never mix."""
    value: dict[str, Any] = {
        "model_sha256": model_sha256,
        "layer": layer,
        "dimension": dimension,
        "normalized": normalized,
    }
    if modality is not None:
        value["modality"] = modality
    return value


def plan(
    surface_request: SurfaceRequest,
    request: EmbeddingRequest,
    entries: Sequence[tuple[Any, dict[str, Any] | None] | str],
    rep: dict[str, Any],
) -> SurfacePlan[Any]:
    """A plan from one ``(item, usage)`` or item error code per input, in input order.

    Items are whatever the family runs (anything with ``ids`` and ``cache_key``).
    """
    runnable: list[Any] = []
    slots: list[int | str] = []
    usages: list[dict[str, Any] | None] = []
    for entry in entries:
        if isinstance(entry, str):
            slots.append(entry)
            usages.append(None)
        else:
            item, usage = entry
            slots.append(len(runnable))
            usages.append(usage)
            runnable.append(item)
    tokens = sum(len(item.ids) for item in runnable)
    return SurfacePlan(
        surface_request.surface,
        runnable,
        tokens,
        EmbeddingPlanState(request, slots, usages, rep),
    )


def finish(
    plan: SurfacePlan[Any],
    results: Any,
    view: Callable[[Any], Sequence[float]] | None = None,
) -> dict[str, Any]:
    """The OpenAI list: failed inputs carry ``error`` in place, plus ``meta.representation``.

    ``view`` turns an item's result into the requested vector (for example a
    Matryoshka view of a cached full-width embedding); results are shared with
    the result cache and never mutated.
    """
    state: EmbeddingPlanState = plan.state
    data = []
    for index, (slot, usage) in enumerate(zip(state.slots, state.usages, strict=True)):
        entry: dict[str, Any] = {"object": "embedding", "index": index}
        if isinstance(slot, str):
            entry["error"] = slot
        else:
            value = DEADLINE if results is DEADLINE else results[slot]
            if value is DEADLINE:
                entry["error"] = DEADLINE_EXCEEDED
            elif value is None:
                entry["error"] = INVALID_MODEL_OUTPUT
            else:
                vector = view(value) if view is not None else value
                entry["embedding"] = encode_vector(vector, state.request.encoding)
            if usage is not None:
                entry["input"] = dict(usage)
        data.append(entry)
    tokens = plan.input_tokens
    return {
        "object": "list",
        "data": data,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        "meta": {"representation": dict(state.representation)},
    }
