"""Bounded-memory, deterministic long-context packing primitives."""

from __future__ import annotations

import hashlib
import sys
from array import array
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_DOCUMENT_TOKENS = 1024 * 1024
UINT32_BYTES = 4


@dataclass
class PackingStats:
    """Mutable audit counters for one language stream."""

    documents_consumed: int = 0
    characters_consumed: int = 0
    source_tokens_consumed: int = 0
    separators_inserted: int = 0
    sequences_emitted: int = 0


def stable_language_quotas(
    languages: Sequence[str], target_sequence_count: int
) -> dict[str, int]:
    """Allocate an exact total using stable list order for the remainder."""
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("languages must be a non-empty ordered set")
    if target_sequence_count < len(languages):
        raise ValueError(
            "target_sequence_count must allocate at least one sequence per language"
        )
    base, remainder = divmod(target_sequence_count, len(languages))
    return {
        language: base + (index < remainder) for index, language in enumerate(languages)
    }


def _encode_document(
    tokenizer: Any, text: str, *, max_document_tokens: int
) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        raise TypeError("document tokenizer unexpectedly returned a batch")
    if len(token_ids) > max_document_tokens:
        raise ValueError(
            f"document token count {len(token_ids)} exceeds cap {max_document_tokens}"
        )
    return [int(token_id) for token_id in token_ids]


def resolve_document_separator_id(tokenizer: Any) -> int:
    """Use a declared separator token; never guess a padding token."""
    for attribute in ("sep_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            return int(token_id)
    raise ValueError("tokenizer must define sep_token_id or eos_token_id")


def real_token_ids(
    input_ids: Sequence[int], attention_mask: Sequence[int] | None
) -> list[int]:
    """Return only attended IDs so padding cannot masquerade as context."""
    if attention_mask is None:
        return [int(token_id) for token_id in input_ids]
    if len(input_ids) != len(attention_mask):
        raise ValueError("input_ids and attention_mask lengths differ")
    return [
        int(token_id)
        for token_id, attended in zip(input_ids, attention_mask, strict=True)
        if int(attended) != 0
    ]


def _emit_ready_sequences(
    payload: list[int],
    tokenizer: Any,
    *,
    language: str,
    payload_length: int,
    target_length: int,
    sequence_quota: int,
    audit: PackingStats,
) -> Iterator[dict[str, Any]]:
    while len(payload) >= payload_length and audit.sequences_emitted < sequence_quota:
        chunk = payload[:payload_length]
        del payload[:payload_length]
        input_ids = list(tokenizer.build_inputs_with_special_tokens(chunk))
        if len(input_ids) != target_length:
            raise RuntimeError(
                "tokenizer special-token contract changed output length: "
                f"{len(input_ids)} != {target_length}"
            )
        audit.sequences_emitted += 1
        yield {
            "input_ids": input_ids,
            "attention_mask": [1] * target_length,
            "language": language,
        }


def iter_packed_sequences(
    documents: Iterable[str],
    tokenizer: Any,
    *,
    language: str,
    target_length: int,
    sequence_quota: int,
    stats: PackingStats | None = None,
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
    max_document_tokens: int = DEFAULT_MAX_DOCUMENT_TOKENS,
) -> Iterator[dict[str, Any]]:
    """Pack source-order documents into exact, unpadded training sequences.

    Memory is bounded by one source document, one tokenizer result, and one
    target-length payload buffer.  Documents are never shuffled.  A declared
    SEP/EOS token marks every document boundary, including boundaries that
    happen to start a new output sequence.
    """
    if sequence_quota <= 0:
        raise ValueError("sequence_quota must be positive")
    if max_document_bytes <= 0 or max_document_tokens <= 0:
        raise ValueError("document byte and token caps must be positive")
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    payload_length = target_length - special_count
    if payload_length <= 1:
        raise ValueError("target_length leaves no room for document tokens")

    separator_id = resolve_document_separator_id(tokenizer)
    audit = stats if stats is not None else PackingStats()
    payload: list[int] = []

    document_iterator = iter(documents)
    while audit.sequences_emitted < sequence_quota:
        try:
            document = next(document_iterator)
        except StopIteration:
            break
        if not isinstance(document, str):
            raise TypeError(f"expected text document, got {type(document).__name__}")
        document_bytes = len(document.encode("utf-8"))
        if document_bytes > max_document_bytes:
            raise ValueError(
                f"document byte count {document_bytes} exceeds cap {max_document_bytes}"
            )

        audit.documents_consumed += 1
        audit.characters_consumed += len(document)
        token_ids = _encode_document(
            tokenizer, document, max_document_tokens=max_document_tokens
        )
        audit.source_tokens_consumed += len(token_ids)

        offset = 0
        while offset < len(token_ids) and audit.sequences_emitted < sequence_quota:
            available = payload_length - len(payload)
            payload.extend(token_ids[offset : offset + available])
            offset += available
            yield from _emit_ready_sequences(
                payload,
                tokenizer,
                language=language,
                payload_length=payload_length,
                target_length=target_length,
                sequence_quota=sequence_quota,
                audit=audit,
            )

        # A trailing separator closes a document that was actually consumed.
        # If it fills the final quota it is a terminal boundary, never a marker
        # created by pre-reading and discarding the next document.
        if audit.sequences_emitted < sequence_quota:
            payload.append(separator_id)
            audit.separators_inserted += 1
            yield from _emit_ready_sequences(
                payload,
                tokenizer,
                language=language,
                payload_length=payload_length,
                target_length=target_length,
                sequence_quota=sequence_quota,
                audit=audit,
            )

    if audit.sequences_emitted != sequence_quota:
        raise RuntimeError(
            f"source stream ended after {audit.sequences_emitted}/{sequence_quota} "
            f"complete {target_length}-token sequences for {language}"
        )


def update_token_digest(digest: Any, token_ids: Iterable[int]) -> None:
    """Hash token IDs with a fixed-width representation."""
    try:
        values = array("I", (int(token_id) for token_id in token_ids))
    except OverflowError as error:
        raise ValueError("token ID is outside uint32 range") from error
    if values.itemsize != UINT32_BYTES:
        raise RuntimeError("platform unsigned-int width is not 32 bits")
    if sys.byteorder != "little":
        values.byteswap()
    digest.update(values.tobytes())


def new_token_digest() -> Any:
    """Return the named digest constructor used by the packing manifest."""
    return hashlib.sha256()
