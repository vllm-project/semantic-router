"""Disk-backed materialization for the exact multilingual foundation corpus."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .cc100_stream import CC100SourceStats, iter_cc100_documents
from .foundation_integrity import (
    ATTENTION_POLICY,
    CC100_UNDERLYING_CONTENT,
    COMMON_CRAWL_TERMS_URL,
    FOUNDATION_WRITER_BATCH_SIZE,
    INTEGRITY_LIMIT,
    LICENSE_ACKNOWLEDGEMENT,
    LOADER_REVISION_SCOPE,
    MANIFEST_SCHEMA_VERSION,
    PACKING_ALGORITHM,
    RELEASE_GATE,
    SOURCE_HASH_SCOPE,
    arrow_shard_records,
    canonical_json_bytes,
    write_canonical_manifest,
)
from .foundation_packing import (
    PackingStats,
    iter_packed_sequences,
    new_token_digest,
    resolve_document_separator_id,
    stable_language_quotas,
    update_token_digest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MaterializationSpec:
    dataset_name: str
    dataset_revision: str
    output_dir: str
    languages: list[str]
    target_sequence_count: int
    max_length: int
    model_name: str
    model_revision: str
    source_etags: list[str]
    source_content_lengths: list[int]
    max_document_bytes: int
    max_document_tokens: int
    acknowledge_cc100_license_unknown: bool
    expected_source_prefixes: dict[str, dict[str, Any]] | None
    text_column: str


def _validate_materialization_inputs(
    *,
    dataset_revision: str,
    model_revision: str,
    languages: list[str],
    source_etags: list[str],
    source_content_lengths: list[int],
    expected_source_prefixes: dict[str, dict[str, Any]] | None,
    acknowledge_cc100_license_unknown: bool,
) -> None:
    if not dataset_revision or not model_revision:
        raise ValueError(
            "streaming packing requires pinned dataset and tokenizer revisions"
        )
    if not languages:
        raise ValueError("streaming packing requires an ordered language list")
    if not (len(source_etags) == len(source_content_lengths) == len(languages)):
        raise ValueError("source contracts must align one-to-one with languages")
    if expected_source_prefixes is not None and set(expected_source_prefixes) != set(
        languages
    ):
        raise ValueError("source prefix contract must cover every language exactly")
    if acknowledge_cc100_license_unknown is not True:
        raise ValueError(
            "exact CC-100 packing requires explicit acknowledgement that its "
            "dataset license is not declared"
        )


def _dataset_fingerprint(
    *,
    dataset_name: str,
    dataset_revision: str,
    languages: list[str],
    source_etags: list[str],
    source_content_lengths: list[int],
    target_sequence_count: int,
    max_length: int,
    model_name: str,
    model_revision: str,
    max_document_bytes: int,
    max_document_tokens: int,
) -> str:
    payload = {
        "algorithm": PACKING_ALGORITHM,
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "languages": languages,
        "source_etags": source_etags,
        "source_content_lengths": source_content_lengths,
        "target_sequence_count": target_sequence_count,
        "max_length": max_length,
        "model_name": model_name,
        "model_revision": model_revision,
        "max_document_bytes": max_document_bytes,
        "max_document_tokens": max_document_tokens,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _expected_prefix_field(
    prefixes: dict[str, dict[str, Any]] | None, language: str, field: str
) -> Any:
    return prefixes[language][field] if prefixes is not None else None


def _iter_examples(
    *,
    dataset_name: str,
    languages: list[str],
    tokenizer: Any,
    quotas: dict[str, int],
    stats: dict[str, PackingStats],
    source_stats: dict[str, CC100SourceStats],
    expected_etags: dict[str, str],
    expected_lengths: dict[str, int],
    expected_source_prefixes: dict[str, dict[str, Any]] | None,
    max_length: int,
    max_document_bytes: int,
    max_document_tokens: int,
    digest: Any,
) -> Iterator[dict[str, Any]]:
    if dataset_name != "statmt/cc100":
        raise ValueError(
            "the exact source-order streaming recipe currently supports only "
            "statmt/cc100"
        )
    for language in languages:
        logger.info(
            "Streaming %s[%s] for %s exact sequences",
            dataset_name,
            language,
            quotas[language],
        )
        documents = iter_cc100_documents(
            language,
            expected_etag=expected_etags[language],
            expected_content_length=expected_lengths[language],
            stats=source_stats[language],
            expected_prefix_bytes=_expected_prefix_field(
                expected_source_prefixes, language, "compressed_bytes_read"
            ),
            expected_prefix_sha256=_expected_prefix_field(
                expected_source_prefixes, language, "compressed_prefix_sha256"
            ),
            max_document_bytes=max_document_bytes,
        )
        with contextlib.closing(documents):
            for example in iter_packed_sequences(
                documents,
                tokenizer,
                language=language,
                target_length=max_length,
                sequence_quota=quotas[language],
                stats=stats[language],
                max_document_bytes=max_document_bytes,
                max_document_tokens=max_document_tokens,
            ):
                update_token_digest(digest, example["input_ids"])
                yield example


def _fixed_features(datasets: Any, max_length: int) -> Any:
    return datasets.Features(
        {
            "input_ids": datasets.Sequence(datasets.Value("int32"), length=max_length),
            "attention_mask": datasets.Sequence(
                datasets.Value("int8"), length=max_length
            ),
            "language": datasets.Value("string"),
        }
    )


def _write_arrow_dataset(
    datasets: Any,
    *,
    destination: Path,
    generate_examples: Any,
    features: Any,
    dataset_fingerprint: str,
    target_sequence_count: int,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="mmbert32k-arrow-", dir=destination.parent
    ) as cache_dir:
        dataset = datasets.Dataset.from_generator(
            generate_examples,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=False,
            fingerprint=dataset_fingerprint,
            writer_batch_size=FOUNDATION_WRITER_BATCH_SIZE,
        )
        if len(dataset) != target_sequence_count:
            raise RuntimeError(
                f"packing emitted {len(dataset)} rows, expected {target_sequence_count}"
            )
        dataset.save_to_disk(destination)


def _language_stats_payload(
    languages: list[str], stats: dict[str, PackingStats]
) -> dict[str, dict[str, int]]:
    return {
        language: {
            "documents_consumed": stats[language].documents_consumed,
            "characters_consumed": stats[language].characters_consumed,
            "source_tokens_consumed": stats[language].source_tokens_consumed,
            "separators_inserted": stats[language].separators_inserted,
            "sequences_emitted": stats[language].sequences_emitted,
        }
        for language in languages
    }


def _source_streams_payload(
    languages: list[str],
    source_stats: dict[str, CC100SourceStats],
    *,
    prefix_contract_verified: bool,
) -> dict[str, dict[str, Any]]:
    return {
        language: {
            "url": source_stats[language].url,
            "etag": source_stats[language].etag,
            "etag_strength": "strong",
            "last_modified": source_stats[language].last_modified,
            "content_length": source_stats[language].content_length,
            "compressed_bytes_read": source_stats[language].compressed_bytes_read,
            "compressed_prefix_sha256": source_stats[language].compressed_prefix_sha256,
            "hash_scope": SOURCE_HASH_SCOPE,
            "full_object_sha256": None,
            "prefix_contract_verified": prefix_contract_verified,
        }
        for language in languages
    }


def _manifest_payload(
    *,
    dataset_name: str,
    dataset_revision: str,
    text_column: str,
    model_name: str,
    model_revision: str,
    max_length: int,
    target_sequence_count: int,
    dataset_fingerprint: str,
    languages: list[str],
    quotas: dict[str, int],
    separator_id: int,
    max_document_bytes: int,
    max_document_tokens: int,
    stats: dict[str, PackingStats],
    source_stats: dict[str, CC100SourceStats],
    prefix_contract_verified: bool,
    packed_input_ids_sha256: str,
    arrow_shards: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "algorithm": PACKING_ALGORITHM,
        "source": {
            "repo_id": dataset_name,
            "revision": dataset_revision,
            "split": "train",
            "text_column": text_column,
            "streaming": True,
            "loader_contract_revision": dataset_revision,
            "loader_revision_scope": LOADER_REVISION_SCOPE,
        },
        "tokenizer": {"repo_id": model_name, "revision": model_revision},
        "target_length": max_length,
        "target_sequence_count": target_sequence_count,
        "dataset_fingerprint": dataset_fingerprint,
        "languages": languages,
        "language_quotas": quotas,
        "document_separator_token_id": separator_id,
        "attention_mask_policy": ATTENTION_POLICY,
        "writer_contract": {
            "writer_batch_size": FOUNDATION_WRITER_BATCH_SIZE,
            "features": {
                "input_ids": {"dtype": "int32", "length": max_length},
                "attention_mask": {"dtype": "int8", "length": max_length},
                "language": {"dtype": "string"},
            },
        },
        "document_caps": {
            "max_bytes": max_document_bytes,
            "max_tokens": max_document_tokens,
        },
        "language_stats": _language_stats_payload(languages, stats),
        "source_streams": _source_streams_payload(
            languages,
            source_stats,
            prefix_contract_verified=prefix_contract_verified,
        ),
        "packed_input_ids_sha256": packed_input_ids_sha256,
        "arrow_shards": arrow_shards,
        "data_governance": {
            "declared_dataset_license": None,
            "underlying_content": CC100_UNDERLYING_CONTENT,
            "common_crawl_terms_url": COMMON_CRAWL_TERMS_URL,
            "license_acknowledgement": LICENSE_ACKNOWLEDGEMENT,
            "release_gate": RELEASE_GATE,
            "integrity_limit": INTEGRITY_LIMIT,
        },
    }


def prepare_streaming_packed_dataset(
    *,
    dataset_name: str,
    dataset_revision: str,
    output_dir: str,
    languages: list[str],
    target_sequence_count: int,
    max_length: int,
    model_name: str,
    model_revision: str,
    source_etags: list[str],
    source_content_lengths: list[int],
    max_document_bytes: int,
    max_document_tokens: int,
    acknowledge_cc100_license_unknown: bool,
    expected_source_prefixes: dict[str, dict[str, Any]] | None = None,
    text_column: str = "text",
):
    """Stream source-order documents and materialize exact unpadded packs."""
    spec = _MaterializationSpec(
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        output_dir=output_dir,
        languages=languages,
        target_sequence_count=target_sequence_count,
        max_length=max_length,
        model_name=model_name,
        model_revision=model_revision,
        source_etags=source_etags,
        source_content_lengths=source_content_lengths,
        max_document_bytes=max_document_bytes,
        max_document_tokens=max_document_tokens,
        acknowledge_cc100_license_unknown=acknowledge_cc100_license_unknown,
        expected_source_prefixes=expected_source_prefixes,
        text_column=text_column,
    )
    return _prepare_materialization(spec)


def _prepare_materialization(spec: _MaterializationSpec) -> Any:
    datasets = importlib.import_module("datasets")
    transformers = importlib.import_module("transformers")
    _validate_materialization_inputs(
        dataset_revision=spec.dataset_revision,
        model_revision=spec.model_revision,
        languages=spec.languages,
        source_etags=spec.source_etags,
        source_content_lengths=spec.source_content_lengths,
        expected_source_prefixes=spec.expected_source_prefixes,
        acknowledge_cc100_license_unknown=(spec.acknowledge_cc100_license_unknown),
    )
    destination = Path(spec.output_dir)
    if destination.exists():
        raise FileExistsError(
            f"refusing to mix a new packing run into existing output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec.model_name, revision=spec.model_revision
    )
    tokenizer.model_max_length = spec.max_length
    separator_id = resolve_document_separator_id(tokenizer)
    quotas = stable_language_quotas(spec.languages, spec.target_sequence_count)
    stats = {language: PackingStats() for language in spec.languages}
    source_stats = {language: CC100SourceStats() for language in spec.languages}
    digest = new_token_digest()
    dataset_fingerprint = _dataset_fingerprint(
        dataset_name=spec.dataset_name,
        dataset_revision=spec.dataset_revision,
        languages=spec.languages,
        source_etags=spec.source_etags,
        source_content_lengths=spec.source_content_lengths,
        target_sequence_count=spec.target_sequence_count,
        max_length=spec.max_length,
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        max_document_bytes=spec.max_document_bytes,
        max_document_tokens=spec.max_document_tokens,
    )
    generate_examples = partial(
        _iter_examples,
        dataset_name=spec.dataset_name,
        languages=spec.languages,
        tokenizer=tokenizer,
        quotas=quotas,
        stats=stats,
        source_stats=source_stats,
        expected_etags=dict(zip(spec.languages, spec.source_etags, strict=True)),
        expected_lengths=dict(
            zip(spec.languages, spec.source_content_lengths, strict=True)
        ),
        expected_source_prefixes=spec.expected_source_prefixes,
        max_length=spec.max_length,
        max_document_bytes=spec.max_document_bytes,
        max_document_tokens=spec.max_document_tokens,
        digest=digest,
    )
    _write_arrow_dataset(
        datasets,
        destination=destination,
        generate_examples=generate_examples,
        features=_fixed_features(datasets, spec.max_length),
        dataset_fingerprint=dataset_fingerprint,
        target_sequence_count=spec.target_sequence_count,
    )
    manifest = _manifest_payload(
        dataset_name=spec.dataset_name,
        dataset_revision=spec.dataset_revision,
        text_column=spec.text_column,
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        max_length=spec.max_length,
        target_sequence_count=spec.target_sequence_count,
        dataset_fingerprint=dataset_fingerprint,
        languages=spec.languages,
        quotas=quotas,
        separator_id=separator_id,
        max_document_bytes=spec.max_document_bytes,
        max_document_tokens=spec.max_document_tokens,
        stats=stats,
        source_stats=source_stats,
        prefix_contract_verified=spec.expected_source_prefixes is not None,
        packed_input_ids_sha256=digest.hexdigest(),
        arrow_shards=arrow_shard_records(destination),
    )
    write_canonical_manifest(destination, manifest)
    logger.info("Saved exact packing manifest to %s", destination)
    return datasets.load_from_disk(destination)
