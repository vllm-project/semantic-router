"""Data-only third-party benchmark packs with one fixed install layout."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from cli.evaluation.constants import TRACK_IDS
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contract_validation import validate_portable_id
from cli.evaluation.errors import SuiteStoreError
from cli.evaluation.reporting import TrackID
from cli.evaluation.suite_contract import BenchmarkSourceReceipt
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    REQUIRED_ARTIFACT_ROLES,
    BenchmarkSuiteInstallRequest,
    SuiteArtifactInstall,
)
from cli.evaluation.suite_store_cas import SuiteCAS

BENCHMARK_PACK_VERSION = "evaluation-benchmark-pack.v1"
BENCHMARK_PACK_MANIFEST = "benchmark.yaml"
BENCHMARK_PACK_BUNDLE = "bundle"
_MAX_MANIFEST_BYTES = 64 * 1024


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise SuiteStoreError("benchmark pack keys must be strings")
        if key in mapping:
            raise SuiteStoreError(f"benchmark pack contains duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class BenchmarkPack(StrictModel):
    """Portable metadata for normalized evidence; never executable code."""

    schema_version: Literal[BENCHMARK_PACK_VERSION] = BENCHMARK_PACK_VERSION
    id: str
    benchmark_id: str = Field(max_length=96)
    name: str = Field(min_length=1, max_length=160)
    decision_unit: str = Field(min_length=1, max_length=256)
    action_space: str = Field(min_length=1, max_length=256)
    track_ids: tuple[TrackID, ...] = Field(min_length=1)
    split_protocol: str = Field(min_length=1, max_length=1024)
    arm_ids: tuple[str, ...] = ()
    data_classification: Literal["public", "internal", "confidential", "restricted"]
    redistribution: Literal["allowed", "metadata_only", "prohibited"]
    limitations: tuple[str, ...] = Field(min_length=1)

    _portable_ids = field_validator("id", "benchmark_id")(validate_portable_id)

    @field_validator("name", "decision_unit", "action_space", "split_protocol")
    @classmethod
    def trimmed_text(cls, value: str) -> str:
        if value.strip() != value:
            raise ValueError("must be trimmed")
        return value

    @field_validator("arm_ids")
    @classmethod
    def unique_arm_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("arm ids must be unique")
        for arm_id in value:
            validate_portable_id(arm_id)
        return value

    @field_validator("limitations")
    @classmethod
    def meaningful_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or item.strip() != item for item in value):
            raise ValueError("limitations must be trimmed and non-empty")
        return value

    @model_validator(mode="after")
    def canonical_tracks(self) -> BenchmarkPack:
        if len(self.track_ids) != len(set(self.track_ids)):
            raise ValueError("track ids must be unique")
        canonical = tuple(track for track in TRACK_IDS if track in self.track_ids)
        if self.track_ids != canonical:
            raise ValueError("track ids must use canonical catalog order")
        return self


def _read_manifest(path: Path) -> dict[str, Any]:
    descriptor = SuiteCAS.open_readonly(path)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_size <= 0 or metadata.st_size > _MAX_MANIFEST_BYTES:
            raise SuiteStoreError("benchmark pack manifest exceeds its size limit")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise SuiteStoreError("benchmark pack manifest changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        text = content.decode("utf-8")
        for token in yaml.scan(text):
            if isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)):
                raise SuiteStoreError("benchmark pack aliases are not supported")
        payload = yaml.load(text, Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SuiteStoreError("benchmark pack manifest is invalid YAML") from exc
    if not isinstance(payload, dict):
        raise SuiteStoreError("benchmark pack manifest must be an object")
    return payload


def load_benchmark_pack(pack_root: str | Path) -> BenchmarkPack:
    root = Path(pack_root).expanduser()
    if root.is_symlink():
        raise SuiteStoreError("benchmark pack root must not be a symlink")
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise SuiteStoreError("benchmark pack root is missing") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise SuiteStoreError("benchmark pack root must be a directory")
    payload = _read_manifest(root / BENCHMARK_PACK_MANIFEST)
    try:
        return BenchmarkPack.model_validate(payload)
    except ValueError as exc:
        raise SuiteStoreError(
            "benchmark pack manifest does not match its contract"
        ) from exc


def _bundle_inventory(bundle_root: Path) -> tuple[SuiteArtifactInstall, ...]:
    root = SuiteCAS.safe_bundle_root(bundle_root)
    allowed_files = {layout[0] for layout in ARTIFACT_ROLE_LAYOUT.values()}
    allowed_directories = {
        str(Path(relative_path).parent) for relative_path in allowed_files
    }
    observed_files: set[str] = set()
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SuiteStoreError("benchmark pack bundle cannot contain symlinks")
        if stat.S_ISDIR(metadata.st_mode):
            if relative not in allowed_directories:
                raise SuiteStoreError(
                    f"benchmark pack bundle contains unknown directory {relative!r}"
                )
            continue
        if not stat.S_ISREG(metadata.st_mode) or relative not in allowed_files:
            raise SuiteStoreError(
                f"benchmark pack bundle contains unknown file {relative!r}"
            )
        observed_files.add(relative)

    artifacts: list[SuiteArtifactInstall] = []
    for role, (relative_path, media_type, _) in ARTIFACT_ROLE_LAYOUT.items():
        if relative_path not in observed_files:
            continue
        digest, size = SuiteCAS.stream_digest_file(root / relative_path)
        artifacts.append(
            SuiteArtifactInstall(
                role=role,
                relative_path=relative_path,
                digest=digest,
                size_bytes=size,
                media_type=media_type,
            )
        )
    missing = REQUIRED_ARTIFACT_ROLES.difference(
        artifact.role for artifact in artifacts
    )
    if missing:
        raise SuiteStoreError(
            "benchmark pack is missing required artifacts: "
            + ", ".join(sorted(missing))
        )
    return tuple(artifacts)


def _jsonl_record_count(path: Path) -> int:
    descriptor = SuiteCAS.open_readonly(path)
    count = 0
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            count += chunk.count(b"\n")
    finally:
        os.close(descriptor)
    return count


def build_pack_install_request(
    pack: BenchmarkPack,
    pack_root: str | Path,
    receipt: BenchmarkSourceReceipt,
) -> tuple[BenchmarkSuiteInstallRequest, Path]:
    root = Path(pack_root).expanduser().resolve(strict=True)
    bundle = root / BENCHMARK_PACK_BUNDLE
    artifacts = _bundle_inventory(bundle)
    case_count = _jsonl_record_count(bundle / "visible/cases.jsonl")
    return (
        BenchmarkSuiteInstallRequest(
            id=pack.id,
            name=pack.name,
            adapter_id=pack.benchmark_id,
            source_receipt=receipt,
            decision_unit=pack.decision_unit,
            action_space=pack.action_space,
            track_ids=pack.track_ids,
            normalization_origin="user_provided_import",
            split_protocol=pack.split_protocol,
            case_count=case_count,
            arm_ids=pack.arm_ids,
            data_classification=pack.data_classification,
            redistribution=pack.redistribution,
            artifacts=artifacts,
            limitations=pack.limitations,
        ),
        bundle,
    )
