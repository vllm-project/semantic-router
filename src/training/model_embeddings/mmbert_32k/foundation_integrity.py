"""Packing-manifest integrity validation and training receipts."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 2
MANIFEST_NAME = "packing_manifest.json"
MANIFEST_DIGEST_NAME = "packing_manifest.sha256"
PACKING_ALGORITHM = "source-order-document-separator-fixed-length-v1"
ATTENTION_POLICY = "all-real-tokens-no-padding"
FOUNDATION_WRITER_BATCH_SIZE = 16
LOADER_REVISION_SCOPE = "loader-code-only-not-source-content"
SOURCE_HASH_SCOPE = "consumed-compressed-prefix-only"
CC100_UNDERLYING_CONTENT = "Common Crawl-derived web content"
COMMON_CRAWL_TERMS_URL = "https://commoncrawl.org/terms-of-use"
RELEASE_GATE = "blocked-pending-data-governance-review"
INTEGRITY_LIMIT = (
    "loader revision pins code; ETag and Content-Length identify the remote "
    "object; SHA-256 covers only the bytes consumed by this run"
)
LICENSE_ACKNOWLEDGEMENT = {
    "cc100_dataset_license_unknown": True,
    "common_crawl_terms_review_required": True,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize a manifest or receipt canonically."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without materializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def arrow_shard_records(dataset_path: Path) -> list[dict[str, Any]]:
    """Return deterministic size/hash records for every Arrow shard."""
    shards = sorted(dataset_path.glob("*.arrow"))
    if not shards:
        raise RuntimeError(f"no Arrow shards found under {dataset_path}")
    return [
        {
            "path": shard.name,
            "size_bytes": shard.stat().st_size,
            "sha256": sha256_file(shard),
        }
        for shard in shards
    ]


def write_canonical_manifest(dataset_path: Path, payload: dict[str, Any]) -> str:
    """Write canonical manifest bytes and their detached SHA-256 sidecar."""
    raw = canonical_json_bytes(payload)
    manifest_path = dataset_path / MANIFEST_NAME
    manifest_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (dataset_path / MANIFEST_DIGEST_NAME).write_text(
        f"{digest}  {MANIFEST_NAME}\n", encoding="utf-8"
    )
    return digest


def _require_fixed_feature(dataset: Any, name: str, length: int, dtype: str) -> None:
    features = getattr(dataset, "features", None)
    if features is None or name not in features:
        raise RuntimeError(f"dataset is missing fixed feature {name!r}")
    feature = features[name]
    if getattr(feature, "length", None) != length:
        raise RuntimeError(f"dataset feature {name!r} is not fixed at {length}")
    value_feature = getattr(feature, "feature", None)
    if getattr(value_feature, "dtype", None) != dtype:
        raise RuntimeError(f"dataset feature {name!r} must use {dtype}")


def _require_scalar_feature(dataset: Any, name: str, dtype: str) -> None:
    features = getattr(dataset, "features", None)
    if features is None or getattr(features.get(name), "dtype", None) != dtype:
        raise RuntimeError(f"dataset feature {name!r} must use {dtype}")


def _validate_attention_values(dataset: Any) -> None:
    try:
        pc = importlib.import_module("pyarrow.compute")

        column = dataset.data.column("attention_mask")
        flattened = pc.list_flatten(column)
        all_real = pc.all(pc.equal(flattened, 1)).as_py()
    except Exception as error:
        raise RuntimeError("could not validate Arrow attention-mask values") from error
    if all_real is not True:
        raise RuntimeError("attention_mask contains padding or non-one values")


def _parse_manifest(dataset_path: Path) -> tuple[dict[str, Any], str]:
    manifest_path = dataset_path / MANIFEST_NAME
    digest_path = dataset_path / MANIFEST_DIGEST_NAME
    if not manifest_path.is_file() or not digest_path.is_file():
        raise RuntimeError("packing manifest or detached digest is missing")
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("packing manifest is not valid JSON") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise RuntimeError("packing manifest is not canonical JSON")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = digest_path.read_text(encoding="utf-8")
    if sidecar != f"{digest}  {MANIFEST_NAME}\n":
        raise RuntimeError("packing manifest digest sidecar does not match")
    return payload, digest


def load_source_prefix_contract(
    dataset_path: str | Path,
    *,
    languages: Iterable[str],
    source_etags: Iterable[str],
    source_content_lengths: Iterable[int],
) -> dict[str, dict[str, Any]]:
    """Load consumed-prefix contracts from a canonical audit materialization."""
    payload, _digest = _parse_manifest(Path(dataset_path))
    language_list = list(languages)
    etag_list = list(source_etags)
    content_lengths = [int(value) for value in source_content_lengths]
    _validate_source_streams(
        payload,
        languages=language_list,
        source_etags=etag_list,
        source_content_lengths=content_lengths,
        require_verified_prefix=False,
    )
    return {
        language: {
            "compressed_bytes_read": payload["source_streams"][language][
                "compressed_bytes_read"
            ],
            "compressed_prefix_sha256": payload["source_streams"][language][
                "compressed_prefix_sha256"
            ],
        }
        for language in language_list
    }


def _validate_source_streams(
    payload: dict[str, Any],
    *,
    languages: list[str],
    source_etags: list[str],
    source_content_lengths: list[int],
    require_verified_prefix: bool = True,
) -> None:
    if not (
        languages
        and len(languages) == len(source_etags) == len(source_content_lengths)
        and len(set(languages)) == len(languages)
    ):
        raise RuntimeError("expected source-contract lists are not aligned")
    if payload.get("languages") != languages:
        raise RuntimeError("packing language order does not match training contract")
    streams = payload.get("source_streams")
    if not isinstance(streams, dict) or set(streams) != set(languages):
        raise RuntimeError("packing source streams do not match languages")
    for language, etag, content_length in zip(
        languages, source_etags, source_content_lengths, strict=True
    ):
        _validate_source_stream(
            streams[language],
            language=language,
            etag=etag,
            content_length=content_length,
            require_verified_prefix=require_verified_prefix,
        )


def _validate_source_stream(
    stream: Any,
    *,
    language: str,
    etag: str,
    content_length: int,
    require_verified_prefix: bool,
) -> None:
    if not isinstance(stream, dict):
        raise RuntimeError(f"packing source stream is invalid for {language}")
    if not etag or etag.strip().startswith("W/"):
        raise RuntimeError(f"expected source ETag is not strong for {language}")
    if stream.get("etag") != etag or stream.get("content_length") != content_length:
        raise RuntimeError(f"packing source contract changed for {language}")
    if stream.get("etag_strength") != "strong":
        raise RuntimeError(f"packing source ETag strength changed for {language}")
    expected_url = f"https://data.statmt.org/cc-100/{language}.txt.xz"
    if stream.get("url") != expected_url:
        raise RuntimeError(f"packing source URL changed for {language}")
    prefix_bytes = stream.get("compressed_bytes_read")
    prefix_sha = stream.get("compressed_prefix_sha256")
    if (
        not isinstance(prefix_bytes, int)
        or prefix_bytes <= 0
        or prefix_bytes > content_length
    ):
        raise RuntimeError(f"packing source prefix length is invalid for {language}")
    if not isinstance(prefix_sha, str) or not _SHA256.fullmatch(prefix_sha):
        raise RuntimeError(f"packing source prefix hash is invalid for {language}")
    if stream.get("hash_scope") != SOURCE_HASH_SCOPE:
        raise RuntimeError(f"packing source hash scope changed for {language}")
    if stream.get("full_object_sha256") is not None:
        raise RuntimeError(f"unexpected unverified full-object hash for {language}")
    if require_verified_prefix and stream.get("prefix_contract_verified") is not True:
        raise RuntimeError(
            f"packing source prefix lacks an external replay contract for {language}"
        )


def _validate_manifest_identity(
    payload: dict[str, Any],
    dataset: Any,
    *,
    expected_source_repo_id: str,
    expected_dataset_revision: str,
    expected_tokenizer_repo_id: str,
    expected_tokenizer_revision: str,
    expected_target_length: int,
    expected_sequence_count: int,
) -> None:
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unsupported packing manifest schema")
    if payload.get("algorithm") != PACKING_ALGORITHM:
        raise RuntimeError("unsupported packing algorithm")
    expected_source = {
        "repo_id": expected_source_repo_id,
        "revision": expected_dataset_revision,
        "split": "train",
        "text_column": "text",
        "streaming": True,
        "loader_contract_revision": expected_dataset_revision,
        "loader_revision_scope": LOADER_REVISION_SCOPE,
    }
    if payload.get("source") != expected_source:
        raise RuntimeError("packing source repository or loader revision changed")
    expected_tokenizer = {
        "repo_id": expected_tokenizer_repo_id,
        "revision": expected_tokenizer_revision,
    }
    if payload.get("tokenizer") != expected_tokenizer:
        raise RuntimeError("packing tokenizer contract changed")
    if payload.get("target_length") != expected_target_length:
        raise RuntimeError("packing target length changed")
    if payload.get("target_sequence_count") != expected_sequence_count:
        raise RuntimeError("packing sequence count changed")
    if len(dataset) != expected_sequence_count:
        raise RuntimeError("Arrow row count does not match packing contract")
    if payload.get("attention_mask_policy") != ATTENTION_POLICY:
        raise RuntimeError("packing attention policy changed")


def _validate_writer_contract(
    payload: dict[str, Any],
    dataset: Any,
    *,
    expected_target_length: int,
    expected_max_document_bytes: int,
    expected_max_document_tokens: int,
    verify_attention_values: bool,
) -> None:
    expected_writer = {
        "writer_batch_size": FOUNDATION_WRITER_BATCH_SIZE,
        "features": {
            "input_ids": {"dtype": "int32", "length": expected_target_length},
            "attention_mask": {"dtype": "int8", "length": expected_target_length},
            "language": {"dtype": "string"},
        },
    }
    if payload.get("writer_contract") != expected_writer:
        raise RuntimeError("packing Arrow writer contract changed")
    expected_caps = {
        "max_bytes": expected_max_document_bytes,
        "max_tokens": expected_max_document_tokens,
    }
    if any(value <= 0 for value in expected_caps.values()):
        raise RuntimeError("expected document caps must be positive")
    if payload.get("document_caps") != expected_caps:
        raise RuntimeError("packing document caps changed")
    separator_id = payload.get("document_separator_token_id")
    if not isinstance(separator_id, int) or separator_id < 0:
        raise RuntimeError("packing document separator is missing or invalid")
    _require_fixed_feature(dataset, "input_ids", expected_target_length, "int32")
    _require_fixed_feature(dataset, "attention_mask", expected_target_length, "int8")
    _require_scalar_feature(dataset, "language", "string")
    if verify_attention_values:
        _validate_attention_values(dataset)


def _validate_digests(payload: dict[str, Any]) -> None:
    if not _SHA256.fullmatch(str(payload.get("packed_input_ids_sha256", ""))):
        raise RuntimeError("packing token digest is missing or invalid")
    if not _SHA256.fullmatch(str(payload.get("dataset_fingerprint", ""))):
        raise RuntimeError("packing dataset fingerprint is missing or invalid")


def _validate_governance(
    payload: dict[str, Any], *, acknowledge_cc100_license_unknown: bool
) -> dict[str, Any]:
    if acknowledge_cc100_license_unknown is not True:
        raise RuntimeError(
            "explicit acknowledgement of the unknown CC-100 dataset license is required"
        )
    expected = {
        "declared_dataset_license": None,
        "underlying_content": CC100_UNDERLYING_CONTENT,
        "common_crawl_terms_url": COMMON_CRAWL_TERMS_URL,
        "license_acknowledgement": LICENSE_ACKNOWLEDGEMENT,
        "release_gate": RELEASE_GATE,
        "integrity_limit": INTEGRITY_LIMIT,
    }
    if payload.get("data_governance") != expected:
        raise RuntimeError("packing data-governance release gate changed")
    return expected


def _validate_language_accounting(
    payload: dict[str, Any], *, languages: list[str], expected_sequence_count: int
) -> None:
    quotient, remainder = divmod(expected_sequence_count, len(languages))
    expected_quotas = {
        language: quotient + (index < remainder)
        for index, language in enumerate(languages)
    }
    quotas = payload.get("language_quotas")
    stats = payload.get("language_stats")
    if quotas != expected_quotas:
        raise RuntimeError("packing language quotas are invalid")
    if not isinstance(stats, dict) or set(stats) != set(languages):
        raise RuntimeError("packing language statistics are missing")
    required_counters = {
        "documents_consumed",
        "characters_consumed",
        "source_tokens_consumed",
        "separators_inserted",
        "sequences_emitted",
    }
    for language in languages:
        language_stats = stats[language]
        if set(language_stats) != required_counters or not all(
            isinstance(language_stats[name], int) and language_stats[name] >= 0
            for name in required_counters
        ):
            raise RuntimeError(f"packing statistics are invalid for {language}")
        if language_stats["sequences_emitted"] != quotas[language]:
            raise RuntimeError(f"packing quota statistics changed for {language}")


def _validate_arrow_shards(root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    expected_shards = payload.get("arrow_shards")
    if not isinstance(expected_shards, list) or not expected_shards:
        raise RuntimeError("packing Arrow shard records are missing")
    actual_shards = arrow_shard_records(root)
    if actual_shards != expected_shards:
        raise RuntimeError("packing Arrow shard size or SHA-256 changed")
    return actual_shards


def _source_content_receipt(
    payload: dict[str, Any], languages: list[str]
) -> dict[str, dict[str, Any]]:
    streams = payload["source_streams"]
    return {
        language: {
            "etag": streams[language]["etag"],
            "etag_strength": "strong",
            "content_length": streams[language]["content_length"],
            "compressed_bytes_read": streams[language]["compressed_bytes_read"],
            "compressed_prefix_sha256": streams[language]["compressed_prefix_sha256"],
            "hash_scope": SOURCE_HASH_SCOPE,
        }
        for language in languages
    }


def validate_packing_manifest(
    dataset_path: str | Path,
    dataset: Any,
    *,
    expected_source_repo_id: str,
    expected_dataset_revision: str,
    expected_tokenizer_repo_id: str,
    expected_tokenizer_revision: str,
    expected_target_length: int,
    expected_sequence_count: int,
    expected_languages: Iterable[str],
    expected_source_etags: Iterable[str],
    expected_source_content_lengths: Iterable[int],
    expected_max_document_bytes: int,
    expected_max_document_tokens: int,
    acknowledge_cc100_license_unknown: bool,
    verify_attention_values: bool = True,
) -> dict[str, Any]:
    """Validate the complete packing handoff before constructing a loader."""
    root = Path(dataset_path)
    if not root.is_dir():
        raise RuntimeError("foundation training requires a local packed dataset")
    payload, manifest_sha256 = _parse_manifest(root)
    _validate_manifest_identity(
        payload,
        dataset,
        expected_source_repo_id=expected_source_repo_id,
        expected_dataset_revision=expected_dataset_revision,
        expected_tokenizer_repo_id=expected_tokenizer_repo_id,
        expected_tokenizer_revision=expected_tokenizer_revision,
        expected_target_length=expected_target_length,
        expected_sequence_count=expected_sequence_count,
    )
    _validate_writer_contract(
        payload,
        dataset,
        expected_target_length=expected_target_length,
        expected_max_document_bytes=expected_max_document_bytes,
        expected_max_document_tokens=expected_max_document_tokens,
        verify_attention_values=verify_attention_values,
    )
    _validate_digests(payload)
    governance = _validate_governance(
        payload,
        acknowledge_cc100_license_unknown=acknowledge_cc100_license_unknown,
    )
    languages = list(expected_languages)
    source_etags = list(expected_source_etags)
    source_content_lengths = [int(value) for value in expected_source_content_lengths]
    _validate_source_streams(
        payload,
        languages=languages,
        source_etags=source_etags,
        source_content_lengths=source_content_lengths,
    )
    _validate_language_accounting(
        payload,
        languages=languages,
        expected_sequence_count=expected_sequence_count,
    )
    actual_shards = _validate_arrow_shards(root, payload)
    return {
        "packing_manifest_sha256": manifest_sha256,
        "arrow_shards": actual_shards,
        "source_content_receipt": _source_content_receipt(payload, languages),
        "data_governance": governance,
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "target_sequence_count": expected_sequence_count,
        "target_length": expected_target_length,
    }


def write_training_receipt(
    output_dir: str | Path,
    *,
    validation: dict[str, Any],
    model_repo_id: str,
    model_revision: str,
    dataset_revision: str,
    status: str,
    optimizer_steps: int,
) -> None:
    """Write a canonical receipt binding output state to validated input data."""
    payload = {
        "schema_version": 1,
        "status": status,
        "model": {"repo_id": model_repo_id, "revision": model_revision},
        "dataset_loader_revision": dataset_revision,
        "packing": validation,
        "optimizer_steps": optimizer_steps,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "training_receipt.json").write_bytes(canonical_json_bytes(payload))
