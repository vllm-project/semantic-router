"""Typed extension registry for evaluation evidence executors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.contracts import RunManifest
from cli.evaluation.evidence import ExecutionRecord, RoutingDiagnostic
from cli.evaluation.execution_contract import EvaluationInputs, ExecutionPlan
from cli.evaluation.executor_contracts import ExecutorContract
from cli.evaluation.store import ArtifactStore

if TYPE_CHECKING:
    from cli.evaluation.suite_store import NormalizedSuiteStore


@dataclass(frozen=True)
class CollectedEvidence:
    inputs: EvaluationInputs
    visible_ref: ArtifactRef
    grading_ref: ArtifactRef
    fixture_ref: ArtifactRef | None
    records: list[ExecutionRecord]
    discovered_entrypoints: tuple[str, ...]
    routing_traces: tuple[RoutingDiagnostic, ...]


class EvidenceExecutor(Protocol):
    contract: ExecutorContract

    def collect(
        self,
        manifest: RunManifest,
        store: ArtifactStore,
        plan: ExecutionPlan,
        suite_store: NormalizedSuiteStore | None,
    ) -> CollectedEvidence: ...


class ExecutorRegistry:
    """Immutable executor lookup that rejects duplicate or unknown identities."""

    def __init__(self, executors: Iterable[EvidenceExecutor]):
        by_id: dict[str, EvidenceExecutor] = {}
        for executor in executors:
            executor_id = executor.contract.id
            if executor_id in by_id:
                raise ValueError(f"duplicate evaluation executor id: {executor_id}")
            by_id[executor_id] = executor
        if not by_id:
            raise ValueError("evaluation executor registry cannot be empty")
        self._by_id = MappingProxyType(by_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id))

    def require(self, executor_id: str) -> EvidenceExecutor:
        try:
            return self._by_id[executor_id]
        except KeyError as exc:
            raise ValueError(f"unknown evaluation executor: {executor_id}") from exc

    def contract(self, executor_id: str) -> ExecutorContract:
        return self.require(executor_id).contract
