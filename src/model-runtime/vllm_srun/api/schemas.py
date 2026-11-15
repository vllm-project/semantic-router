"""The System One schema a runtime process speaks on ``/v1/systemone`` and ``/v1/systemone/batch``.

``vllm-srun serve --systemone-schema`` names it: ``vllm-sr/5.0``, the default, is the System One standard
plus the vllm-sr extension set (config/schemas/systemone/vllm-sr-5.0.extensions.yaml); ``vllm-sr/4.0`` is a
compatible superset that accepts every form the runtime contract 4.0.0 accepted and every 5.0 form. Either
is read into the runtime's internal request form (the 4.0 forms plus ``roles``), which ``/v1/decisions`` and
``/v1/bundle`` read as they are, and the answers are written back in the client's schema. The Router's
listeners apply the same rules (``systemonecodec`` in the Go module).

Under 5.0 a request is decoded strictly, any member the schema does not define refused with its name; an
object state without ``roles`` gives no part a role (the internal form's ``roles: null``); and assets whose
placeholders lead the state (images, then videos, then one other item) are read in front of that item, as
``images`` and ``videos`` are. Under 4.0 a request passes as it came, and an object state without ``roles``
keeps the implicit roles its keys give it.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

from .systemone_fields import (
    ASSET_FIELDS,
    ASSET_TYPES,
    OPTION_FIELDS,
    QUESTION_FIELDS,
    QUESTION_TYPES,
    REMOVED,
    REQUEST_FIELDS,
)

SCHEMA_50 = "vllm-sr/5.0"
SCHEMA_40 = "vllm-sr/4.0"
SCHEMAS = (SCHEMA_50, SCHEMA_40)
LONG_CONTEXT_HEADER = "typesafe-danger-zone-long-context"
PATH_SYSTEMONE = "/v1/systemone"
PATH_BATCH = "/v1/systemone/batch"
PATH_DECISIONS = "/v1/decisions"
# The standard's status of an overloaded server.
OVERLOADED = 529
UNPROCESSABLE = 422
# Placed media is at least one placeholder followed by the state.
_PLACED_MIN_ITEMS = 2
# How long a client waits before it asks again, by error code.
RETRY_AFTER = {"overloaded": "1", "not_ready": "5"}
_IMPLICIT = object()
_VIDEO_MIMES = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
}
_IMAGE_MIMES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}


class ReadError(ValueError):
    """A request the schema does not accept; 400 invalid_request."""


@dataclass(frozen=True)
class Read:
    """A request read into the internal form, and the names of a batch endpoint's requests."""

    body: dict[str, Any]
    names: list[str] | None = None


def read(schema: str, path: str, body: Any, headers: Any) -> Read:
    """A request on one of the System One paths in the internal form; raises ReadError."""
    long_context = str(headers.get(LONG_CONTEXT_HEADER, "")).strip().lower()
    if long_context not in ("", "true"):
        raise ReadError("TypeSafe-Danger-Zone-Long-Context takes true")
    five = schema == SCHEMA_50
    roles = None if five else _IMPLICIT
    if path == PATH_BATCH:
        internal, names = _batch(body, roles, five)
    elif five and path == PATH_SYSTEMONE:
        internal, names = _alone(body, roles), None
    else:
        if not isinstance(body, dict):
            raise ReadError("the request body must be a JSON object")
        internal, names = body, None
    if long_context == "true":
        options = internal.get("options")
        internal = {
            **internal,
            "options": {
                **(options if isinstance(options, dict) else {}),
                "long_context": True,
            },
        }
    return Read(internal, names)


def _check(obj: dict[str, Any], allowed: tuple[str, ...], level: str, at: str) -> None:
    for key in obj:
        if key in allowed:
            continue
        replaced = REMOVED.get((level, key))
        if replaced:
            raise ReadError(f"{at}{key} is not part of vllm-sr/5.0; use {replaced}")
        raise ReadError(f"{at}{key} is not a field of a vllm-sr/5.0 {level}")


def _options(options: Any) -> dict[str, Any]:
    if not isinstance(options, dict):
        raise ReadError("options must be an object")
    for key in options:
        if key == "long_context":
            raise ReadError(
                "options.long_context is not an option in vllm-sr/5.0; send the request header TypeSafe-Danger-Zone-Long-Context: true"
            )
        if key not in OPTION_FIELDS:
            raise ReadError(f"options.{key} is not an option")
    return dict(options)


def _alone(body: Any, roles: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ReadError("the request must be a JSON object")
    _check(body, REQUEST_FIELDS, "request", "")
    internal: dict[str, Any] = {}
    if "model" in body:
        internal["model"] = body["model"]
    internal.update(_request(body, "", roles, front_media=True))
    if "options" in body:
        internal["options"] = _options(body["options"])
    return internal


def _batch(
    body: Any, roles: Any, front_media: bool
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(body, dict):
        raise ReadError("the request must be a JSON object")
    for key in body:
        if key not in ("model", "requests", "options"):
            raise ReadError(
                f"{key} is not a member of a batch; a batch takes model, requests and options"
            )
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ReadError("model is required")
    requests = body.get("requests")
    if not isinstance(requests, dict) or not requests:
        raise ReadError("requests must be a nonempty object of named requests")
    states: dict[str, Any] = {}
    shared: dict[str, Any] = {}
    for name, item in requests.items():
        at = f"requests.{name}."
        if not isinstance(item, dict):
            raise ReadError(f"requests.{name} must be an object")
        if "options" in item:
            raise ReadError(f"{at}options: a batch's options are the batch's own")
        _check(item, REQUEST_FIELDS, "request", at)
        if "model" in item and item["model"] != model:
            raise ReadError(
                f"{at}model {item['model']!r} is not the batch's model {model!r}"
            )
        entry = _request(item, at, roles, front_media)
        for placeholder, asset in entry.pop("assets", {}).items():
            if placeholder in shared and shared[placeholder] != asset:
                raise ReadError(
                    f"request {name!r} uses placeholder {placeholder!r} for another asset than an earlier request of the batch"
                )
            shared[placeholder] = asset
        states[name] = entry
    internal: dict[str, Any] = {"model": model, "states": states}
    if shared:
        internal["assets"] = shared
    if "options" in body:
        internal["options"] = _options(body["options"])
    return internal, list(requests)


def _request(
    obj: dict[str, Any], at: str, roles: Any, front_media: bool
) -> dict[str, Any]:
    if "state" not in obj:
        raise ReadError(f"{at}state is required")
    state = obj["state"]
    questions = obj.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ReadError(f"{at}questions must be a nonempty object of named questions")
    for question_id, question in questions.items():
        where = f"{at}questions.{question_id}"
        if not isinstance(question, dict):
            raise ReadError(f"{where} must be an object")
        kind = question.get("type")
        if kind is None:
            raise ReadError(f"{where}: type is required")
        if kind not in QUESTION_TYPES:
            raise ReadError(f"{where}: type {kind!r} is not a question type")
        for key in question:
            if key in QUESTION_FIELDS[kind]:
                continue
            replaced = REMOVED.get(("question", key))
            if replaced:
                raise ReadError(
                    f"{where}: {key} is not part of vllm-sr/5.0; use {replaced}"
                )
            raise ReadError(
                f"{where}: {key} is not a field of a vllm-sr/5.0 {kind} question"
            )
    assets = obj.get("assets", {})
    if not isinstance(assets, dict):
        raise ReadError(f"{at}assets must be an object of placeholders to assets")
    for placeholder, asset in assets.items():
        if not isinstance(asset, dict) or set(asset) - set(ASSET_FIELDS):
            raise ReadError(f"{at}assets.{placeholder} takes {', '.join(ASSET_FIELDS)}")
        if asset.get("type", "image") not in ASSET_TYPES:
            raise ReadError(
                f"{at}assets.{placeholder}: type {asset.get('type')!r} is not an asset type"
            )
    entry: dict[str, Any] = {}
    images: list[str] = []
    videos: list[str] = []
    if front_media:
        state, images, videos, assets = _unplace(state, assets, at)
    for placeholder, asset in assets.items():
        if asset.get("type") == "video":
            raise ReadError(
                f"{at}assets.{placeholder}: a video asset goes in front of the state: [image placeholders..., video placeholders..., state]"
            )
    entry["state"] = state
    entry["questions"] = questions
    if assets:
        entry["assets"] = assets
    if images:
        entry["images"] = images
    if videos:
        entry["videos"] = videos
    if "roles" in obj:
        given = obj["roles"]
        if not isinstance(given, dict):
            raise ReadError(f"{at}roles must be an object of state keys to roles")
        if not isinstance(state, dict):
            raise ReadError(f"{at}roles name the keys of an object state")
        for key, role in given.items():
            if key not in state:
                raise ReadError(f"{at}roles.{key} names no key of the state")
            if role not in ("user", "context", "answer"):
                raise ReadError(f"{at}roles.{key} must be user, context or answer")
        entry["roles"] = given
    elif isinstance(state, dict) and roles is not _IMPLICIT:
        entry["roles"] = roles
    return entry


def _unplace(
    state: Any, assets: dict[str, Any], at: str
) -> tuple[Any, list[str], list[str], dict[str, Any]]:
    """Media placed in front of a state: an array of image placeholders, then video placeholders, then exactly
    one other item is that item, read with the media in front of it."""
    if not isinstance(state, list) or len(state) < _PLACED_MIN_ITEMS:
        return state, [], [], assets
    images: list[str] = []
    videos: list[str] = []
    used: set[str] = set()
    lead = 0
    while lead < len(state) - 1:
        name = state[lead]
        if not isinstance(name, str) or name not in assets or name in used:
            break
        asset = assets[name]
        kind = asset.get("type", "image")
        if kind == "image" and videos:
            break
        url = _data_url(kind, asset, f"{at}assets.{name}")
        used.add(name)
        (videos if kind == "video" else images).append(url)
        lead += 1
    if lead == 0 or lead != len(state) - 1:
        return state, [], [], assets
    return (
        state[lead],
        images,
        videos,
        {key: value for key, value in assets.items() if key not in used},
    )


def _data_url(kind: str, asset: dict[str, Any], where: str) -> str:
    data = asset.get("data_base64")
    if not isinstance(data, str) or not data:
        raise ReadError(f"{where}: a placed medium needs its data_base64")
    fmt = str(asset.get("format", "")).lower()
    if kind == "video":
        if fmt not in _VIDEO_MIMES:
            raise ReadError(
                f"{where}: a video asset needs its format: mp4, webm, mov or mkv"
            )
        return f"data:{_VIDEO_MIMES[fmt]};base64,{data}"
    if fmt in _IMAGE_MIMES:
        return f"data:{_IMAGE_MIMES[fmt]};base64,{data}"
    if fmt:
        raise ReadError(f"{where}: an image asset's format is png, jpg or jpeg")
    return f"data:{_sniff_image(data)};base64,{data}"


def _sniff_image(data: str) -> str:
    head = data[:24]
    try:
        raw = base64.b64decode(head[: len(head) // 4 * 4])
    except (binascii.Error, ValueError):
        raw = b""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def answer(read_request: Read | None, response: dict[str, Any]) -> dict[str, Any]:
    """A success in the client's form: a batch endpoint's entries under ``responses``, anything else as it is."""
    if read_request is None or read_request.names is None or "states" not in response:
        return response
    answered = {"model": response.get("model"), "responses": response["states"]}
    for key, value in response.items():
        if key not in ("model", "states"):
            answered[key] = value
    return answered


def error(
    schema: str, path: str, status: int, body: Any, batch: bool = False
) -> tuple[int, Any, dict[str, str]]:
    """An internal error ({error: {...}}) in the client's schema, its status and headers.

    Under vllm-sr/4.0, and on /v1/decisions, 4.0's ``error`` stays beside the standard ``detail`` and the flat
    fields. Under vllm-sr/5.0 the body is ``detail`` (a list on a 422) and the flat ``code``, ``question``,
    ``state``, ``reason`` and ``limit``, and an overloaded server answers 529. A refusal a client may retry
    says when.
    """
    native = body.get("error") if isinstance(body, dict) else None
    if not isinstance(native, dict):
        return status, body, {}
    code = native.get("code", "")
    message = native.get("message") or str(code).replace("_", " ")
    headers = {"Retry-After": RETRY_AFTER[code]} if code in RETRY_AFTER else {}
    flat = {
        key: native[key]
        for key in ("code", "question", "state", "reason", "limit")
        if native.get(key) is not None
    }
    if schema != SCHEMA_50 or path == PATH_DECISIONS:
        return status, {"error": native, "detail": message, **flat}, headers
    if code == "overloaded":
        status = OVERLOADED
    detail: Any = message
    if status == UNPROCESSABLE:
        location: list[str] = ["body"]
        if batch and native.get("state"):
            location += ["requests", native["state"]]
        if native.get("question"):
            location += ["questions", native["question"]]
        detail = [
            {
                "loc": location,
                "msg": message,
                "type": native.get("question_error") or code,
            }
        ]
    return status, {"detail": detail, **flat}, headers


def listing(
    schema: str, cards: list[dict[str, Any]], listed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The listing's ``models``: under vllm-sr/5.0 each decisions model with the extension set's version, its
    features (the question types it answers) and its card; under vllm-sr/4.0 the entries as they are.
    """
    if schema != SCHEMA_50:
        return listed
    by_id = {card.get("id"): card for card in cards}
    entries = []
    for entry in listed:
        card = by_id.get(entry["name"], {})
        entries.append(
            {
                **entry,
                "version": "5.0",
                "features": list(card.get("question_types") or []),
                "card": card,
            }
        )
    return entries
