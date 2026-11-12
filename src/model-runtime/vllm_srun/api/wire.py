"""System One wire forms: compressed bodies and multipart/form-data, read into the canonical request.

A body may be compressed with one layer of ``gzip`` (``x-gzip``) or ``zstd``
(a window of at most 8 MiB); the size limit applies to the body as sent and
again once decoded. Another coding, or more than one, is 415. A
``multipart/form-data`` body carries the JSON request in its first part,
``request``, and each image's bytes in a part of its own that exactly one
asset claims (``assets.<reference>.part``); the claimed bytes replace the
asset's ``part`` in the canonical request, so everything after this module
reads one form.
"""

from __future__ import annotations

import gzip
import io
import sys
import zlib
from email.message import Message
from typing import Any

from ..errors import RequestError, RuntimeServiceError

ZSTD_MAX_WINDOW = 8 << 20
MULTIPART = "multipart/form-data"
# RFC 2046's longest boundary.
MAX_BOUNDARY = 70
REQUEST_PART = "request"
IMAGE_PART_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "application/octet-stream"}
)


def content_encoding(values: list[str]) -> str | None:
    """The one coding of a ``Content-Encoding`` header (``gzip`` or ``zstd``); None for none or identity."""
    codings = [
        coding.strip().lower()
        for value in values
        for coding in value.split(",")
        if coding.strip() and coding.strip().lower() != "identity"
    ]
    if not codings:
        return None
    if len(codings) > 1:
        raise RuntimeServiceError(
            "unsupported_media_type",
            "send the body with at most one Content-Encoding (gzip or zstd)",
            "encoding_unsupported",
        )
    if codings[0] in ("gzip", "x-gzip"):
        return "gzip"
    if codings[0] == "zstd":
        return "zstd"
    raise RuntimeServiceError(
        "unsupported_media_type",
        f"Content-Encoding {codings[0][:32]!r} is not supported: send the body uncompressed, "
        "or compressed with gzip or zstd",
        "encoding_unsupported",
    )


def decode(raw: bytes, coding: str, limit: int) -> bytes:
    """``raw`` decompressed (one layer of ``coding``) into at most ``limit`` bytes."""
    if coding == "gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                body = stream.read(limit + 1)
        except (OSError, EOFError, zlib.error) as exc:
            raise _malformed(coding) from exc
    else:
        body = _zstd(raw, limit)
    if len(body) > limit:
        raise _too_large(limit)
    return body


def _zstd(raw: bytes, limit: int) -> bytes:
    """Every zstd frame of ``raw``, decoded with at most an 8 MiB window into at most ``limit + 1`` bytes."""
    if sys.version_info >= (3, 14):
        from compression.zstd import DecompressionParameter, ZstdDecompressor, ZstdError
    else:
        from backports.zstd import DecompressionParameter, ZstdDecompressor, ZstdError
    chunks: list[bytes] = []
    total = 0
    data = raw
    try:
        while data:
            decoder = ZstdDecompressor(
                options={
                    DecompressionParameter.window_log_max: ZSTD_MAX_WINDOW.bit_length()
                    - 1
                }
            )
            chunk = decoder.decompress(data, max_length=limit + 1 - total)
            while True:
                chunks.append(chunk)
                total += len(chunk)
                if total > limit:
                    return b"".join(chunks)
                if decoder.eof:
                    break
                if decoder.needs_input:
                    raise _malformed("zstd")
                chunk = decoder.decompress(b"", max_length=limit + 1 - total)
            data = decoder.unused_data
    except ZstdError as exc:
        if "memory" in str(exc).lower() or "window" in str(exc).lower():
            raise RuntimeServiceError(
                "invalid_request",
                "zstd request bodies may use a window of at most 8 MiB",
                "encoding_malformed",
            ) from exc
        raise _malformed("zstd") from exc
    return b"".join(chunks)


def _too_large(limit: int) -> RuntimeServiceError:
    return RuntimeServiceError(
        "request_too_large",
        f"the request body exceeds {limit} bytes once decompressed",
    )


def _malformed(coding: str) -> RuntimeServiceError:
    return RuntimeServiceError(
        "invalid_request",
        f"the request body is not valid {coding} data",
        "encoding_malformed",
    )


def boundary(content_type: str | None) -> str | None:
    """The boundary of a ``multipart/form-data`` content type; None for any other body."""
    if not content_type:
        return None
    header = Message()
    header["content-type"] = content_type
    if header.get_content_type() != MULTIPART:
        return None
    value = header.get_param("boundary")
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_BOUNDARY:
        raise RequestError(
            "multipart_malformed", "a multipart/form-data body needs a boundary"
        )
    return value


def multipart(
    raw: bytes, mark: str
) -> tuple[bytes, dict[str, tuple[str | None, bytes]]]:
    """The first part (``request``, the JSON request) and every other part by name: (content type, bytes)."""
    delimiter = b"--" + mark.encode("latin-1")
    position = raw.find(delimiter)
    if position < 0 or (position > 0 and raw[position - 2 : position] != b"\r\n"):
        raise RequestError(
            "multipart_malformed", "the multipart body has no boundary line"
        )
    position += len(delimiter)
    parts: list[tuple[str, str | None, bytes]] = []
    while True:
        if raw[position : position + 2] == b"--":
            break
        end_of_line = raw.find(b"\r\n", position)
        if end_of_line < 0 or raw[position:end_of_line].strip(b" \t"):
            raise RequestError(
                "multipart_malformed", "a multipart boundary line is malformed"
            )
        start = end_of_line + 2
        end = raw.find(b"\r\n" + delimiter, start)
        if end < 0:
            raise RequestError(
                "multipart_malformed", "the multipart body is not terminated"
            )
        parts.append(_part(raw[start:end]))
        position = end + 2 + len(delimiter)
    if not parts or parts[0][0] != REQUEST_PART:
        raise RequestError(
            "multipart_malformed", "the first multipart part must be named request"
        )
    name, content_type, request = parts[0]
    if content_type is not None and content_type != "application/json":
        raise RequestError(
            "multipart_malformed", "the request part must be application/json"
        )
    files: dict[str, tuple[str | None, bytes]] = {}
    for name, content_type, data in parts[1:]:
        if name == REQUEST_PART or name in files:
            raise RequestError(
                "multipart_malformed",
                f"multipart part {name[:64]!r} appears more than once",
            )
        if content_type is not None and content_type not in IMAGE_PART_TYPES:
            raise RequestError(
                "multipart_malformed",
                f"multipart part {name[:64]!r} has content type {content_type[:64]!r}; "
                "image parts are image/png, image/jpeg or application/octet-stream",
            )
        files[name] = (content_type, data)
    return request, files


def _part(data: bytes) -> tuple[str, str | None, bytes]:
    if data.startswith(b"\r\n"):
        head, body = b"", data[2:]
    else:
        split = data.find(b"\r\n\r\n")
        if split < 0:
            raise RequestError(
                "multipart_malformed", "a multipart part has no header end"
            )
        head, body = data[:split], data[split + 4 :]
    headers = Message()
    for line in head.split(b"\r\n") if head else ():
        field, colon, value = line.partition(b":")
        if not colon:
            raise RequestError(
                "multipart_malformed", "a multipart part header is malformed"
            )
        headers[field.decode("latin-1").strip()] = value.decode(
            "utf-8", "replace"
        ).strip()
    if headers.get_content_disposition() != "form-data":
        raise RequestError(
            "multipart_malformed",
            "every multipart part needs Content-Disposition: form-data",
        )
    name = headers.get_param("name", header="content-disposition")
    if not isinstance(name, str) or not name:
        raise RequestError("multipart_malformed", "every multipart part needs a name")
    content_type = headers.get_content_type() if headers.get("content-type") else None
    return name, content_type, body


def attach_parts(body: Any, files: dict[str, tuple[str | None, bytes]]) -> None:
    """Give each asset that names a part that part's bytes; every part must be claimed exactly once."""
    assets = body.get("assets") if isinstance(body, dict) else None
    claimed: set[str] = set()
    if isinstance(assets, dict):
        for reference, entry in assets.items():
            if not isinstance(entry, dict) or "part" not in entry:
                continue
            name = entry["part"]
            if not isinstance(name, str) or name not in files:
                raise RequestError(
                    "multipart_malformed",
                    f"assets[{str(reference)[:64]!r}] names a part the body does not have",
                )
            if name in claimed:
                raise RequestError(
                    "multipart_malformed",
                    f"multipart part {name[:64]!r} is claimed twice",
                )
            claimed.add(name)
            # A data_base64 next to a part is ignored: the claimed bytes are the asset.
            entry.pop("data_base64", None)
            entry.pop("part")
            entry["data"] = files[name][1]
    unclaimed = sorted(set(files) - claimed)
    if unclaimed:
        raise RequestError(
            "multipart_malformed",
            f"multipart part {unclaimed[0][:64]!r} is claimed by no asset",
        )
