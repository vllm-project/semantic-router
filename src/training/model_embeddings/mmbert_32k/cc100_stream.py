"""Standard-library streaming reader for the pinned CC-100 source layout."""

from __future__ import annotations

import hashlib
import io
import lzma
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

CC100_URL_TEMPLATE = "https://data.statmt.org/cc-100/{language}.txt.xz"


@dataclass
class CC100SourceStats:
    """HTTP and byte-level audit data for one partially consumed source."""

    url: str = ""
    etag: str = ""
    last_modified: str = ""
    content_length: int = 0
    compressed_bytes_read: int = 0
    compressed_prefix_sha256: str = ""


def _normalize_etag(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]
    if not normalized or '"' in normalized:
        raise ValueError("ETag must contain one non-empty opaque tag")
    return normalized


def _is_weak_etag(value: str) -> bool:
    return value.strip().startswith("W/")


class _HashingReader:
    def __init__(self, raw: Any, stats: CC100SourceStats):
        self.raw = raw
        self.stats = stats
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.stats.compressed_bytes_read += len(chunk)
        self.digest.update(chunk)
        self.stats.compressed_prefix_sha256 = self.digest.hexdigest()
        return chunk

    @staticmethod
    def readable() -> bool:
        return True

    @staticmethod
    def seekable() -> bool:
        return False


def _validate_expected_contract(
    language: str,
    *,
    expected_etag: str,
    expected_content_length: int,
    expected_prefix_bytes: int | None,
    expected_prefix_sha256: str | None,
    max_document_bytes: int,
) -> str:
    if not expected_etag:
        raise ValueError(f"missing pinned CC-100 ETag for {language}")
    if _is_weak_etag(expected_etag):
        raise ValueError(f"weak expected ETag is not allowed for {language}")
    try:
        normalized_expected_etag = _normalize_etag(expected_etag)
    except ValueError as error:
        raise ValueError(f"invalid expected ETag for {language}") from error
    if expected_content_length <= 0:
        raise ValueError(f"missing positive content length for {language}")
    if max_document_bytes <= 0:
        raise ValueError("max_document_bytes must be positive")
    if (expected_prefix_bytes is None) != (expected_prefix_sha256 is None):
        raise ValueError("prefix bytes and SHA-256 must be configured together")
    return normalized_expected_etag


def _record_response_contract(
    response: Any,
    *,
    language: str,
    expected_etag: str,
    expected_content_length: int,
    url: str,
    stats: CC100SourceStats,
) -> None:
    raw_etag = response.headers.get("ETag", "")
    if _is_weak_etag(raw_etag):
        raise RuntimeError(f"CC-100 {language} returned a weak ETag")
    if not (raw_etag.startswith('"') and raw_etag.endswith('"')):
        raise RuntimeError(f"CC-100 {language} returned a non-strong ETag")
    try:
        observed_etag = _normalize_etag(raw_etag)
    except ValueError as error:
        raise RuntimeError(f"CC-100 {language} returned an invalid ETag") from error
    if observed_etag != expected_etag:
        raise RuntimeError(
            f"CC-100 {language} ETag changed: {observed_etag!r} != {expected_etag!r}"
        )

    stats.url = url
    stats.etag = observed_etag
    stats.last_modified = response.headers.get("Last-Modified", "")
    try:
        stats.content_length = int(response.headers.get("Content-Length", ""))
    except ValueError as error:
        raise RuntimeError(f"CC-100 {language} has invalid Content-Length") from error
    if stats.content_length != expected_content_length:
        raise RuntimeError(
            f"CC-100 {language} Content-Length changed: {stats.content_length} "
            f"!= {expected_content_length}"
        )


def _iter_text_documents(
    text_stream: Any,
    *,
    language: str,
    max_document_bytes: int,
) -> Iterator[str]:
    paragraphs: list[str] = []
    document_bytes = 0
    for line in text_stream:
        paragraph = line.rstrip("\r\n")
        if paragraph:
            paragraph_bytes = len(paragraph.encode("utf-8"))
            document_bytes += paragraph_bytes + bool(paragraphs)
            if document_bytes > max_document_bytes:
                raise ValueError(
                    f"CC-100 {language} document exceeds {max_document_bytes} bytes"
                )
            paragraphs.append(paragraph)
        elif paragraphs:
            yield "\n".join(paragraphs)
            paragraphs = []
            document_bytes = 0
    if paragraphs:
        yield "\n".join(paragraphs)


def _validate_consumed_prefix(
    language: str,
    *,
    stats: CC100SourceStats,
    expected_prefix_bytes: int | None,
    expected_prefix_sha256: str | None,
) -> None:
    if expected_prefix_bytes is None:
        return
    if stats.compressed_bytes_read != expected_prefix_bytes:
        raise RuntimeError(
            f"CC-100 {language} consumed-prefix length changed: "
            f"{stats.compressed_bytes_read} != {expected_prefix_bytes}"
        )
    if stats.compressed_prefix_sha256 != expected_prefix_sha256:
        raise RuntimeError(f"CC-100 {language} consumed-prefix SHA-256 changed")


def iter_cc100_documents(
    language: str,
    *,
    expected_etag: str,
    expected_content_length: int,
    stats: CC100SourceStats,
    expected_prefix_bytes: int | None = None,
    expected_prefix_sha256: str | None = None,
    max_document_bytes: int = 8 * 1024 * 1024,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Iterator[str]:
    """Yield source-order documents separated by CC-100 blank lines."""
    normalized_expected_etag = _validate_expected_contract(
        language,
        expected_etag=expected_etag,
        expected_content_length=expected_content_length,
        expected_prefix_bytes=expected_prefix_bytes,
        expected_prefix_sha256=expected_prefix_sha256,
        max_document_bytes=max_document_bytes,
    )
    url = CC100_URL_TEMPLATE.format(language=language)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "vllm-semantic-router-mmbert32k/1"},
    )
    with opener(request, timeout=60) as response:
        _record_response_contract(
            response,
            language=language,
            expected_etag=normalized_expected_etag,
            expected_content_length=expected_content_length,
            url=url,
            stats=stats,
        )
        compressed = _HashingReader(response, stats)
        try:
            with (
                lzma.LZMAFile(compressed, mode="rb") as decompressed,
                io.TextIOWrapper(decompressed, encoding="utf-8") as text_stream,
            ):
                yield from _iter_text_documents(
                    text_stream,
                    language=language,
                    max_document_bytes=max_document_bytes,
                )
        finally:
            _validate_consumed_prefix(
                language,
                stats=stats,
                expected_prefix_bytes=expected_prefix_bytes,
                expected_prefix_sha256=expected_prefix_sha256,
            )
