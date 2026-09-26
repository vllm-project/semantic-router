"""The narrow contract between a Decision model family and the shared runtime."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from decision_runtime.artifacts import VerifiedArtifact
    from decision_runtime.physical_batching import DecisionRow
    from decision_runtime.runtime_profile import (
        BackendExecutionPolicy,
        RuntimeProfile,
    )


class FamilyLoadError(RuntimeError):
    """Verified release data cannot be loaded by its registered family."""


@dataclass(frozen=True, slots=True)
class FamilyArtifactFiles:
    """Family-owned file selection; shared artifact code verifies every file."""

    required: frozenset[str]
    optional: frozenset[str]


class DecisionFamilyAdapter(Protocol):
    family: str
    manifest_paths: frozenset[str]
    encoded_row_type: type
    inference_workers: int | None

    def artifact_files(
        self, manifest_path: str, available_paths: Collection[str]
    ) -> FamilyArtifactFiles: ...

    def batch_key(self, question_type: str) -> str: ...

    def encode_rows(
        self, rows: tuple[DecisionRow, ...], tokenizer: Any, profile: RuntimeProfile
    ) -> tuple[Any, ...]: ...

    def parse_execution(
        self, value: object
    ) -> Mapping[str, BackendExecutionPolicy]: ...

    def load(
        self,
        artifact: VerifiedArtifact,
        profile: RuntimeProfile,
        backend: str,
        *,
        physical_batch_size: int,
        graph_event_recorder: Any,
    ) -> Any: ...
