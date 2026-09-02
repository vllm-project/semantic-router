from __future__ import annotations

import json
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest
from cli.evaluation.benchmark_registry import (
    ADAPTER_CONTRACT_VERSION,
    get_benchmark_adapter,
    get_benchmark_registry,
)
from cli.evaluation.benchmark_sources import (
    SourceVerificationError,
    verify_benchmark_source,
)
from cli.evaluation.canonical import digest_value
from cli.evaluation.contract_primitives import ArtifactRef
from cli.evaluation.research_benchmark_inventory import RESEARCH_BENCHMARKS_BY_ADAPTER
from cli.evaluation.suite_contract import (
    SUITE_CONTRACT_VERSION,
    BenchmarkSourceReceipt,
    BenchmarkSuiteManifest,
    SuiteArtifactSet,
    qualification_manifest_subject_digest,
)
from cli.evaluation.suite_qualification import (
    NORMALIZED_REPLAY_EXECUTOR_DIGEST,
    BenchmarkQualificationReceipt,
    UnqualifiedBenchmarkEvidence,
)
from pydantic import ValidationError


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(path), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _ref(character: str) -> ArtifactRef:
    return ArtifactRef(
        digest="sha256:" + character * 64,
        media_type="application/x-ndjson",
        size_bytes=1,
    )


def test_registry_covers_every_audited_benchmark_at_an_exact_pin() -> None:
    registry = get_benchmark_registry()
    assert registry.schema_version == ADAPTER_CONTRACT_VERSION
    assert len(registry.adapters) == 13
    assert len({adapter.id for adapter in registry.adapters}) == 13
    assert {adapter.id for adapter in registry.adapters} == {
        "routerarena",
        "routejudge-orbit",
        "coderouterbench",
        "llmrouterbench",
        "routereval",
        "routerbench",
        "xroutebench",
        "twinrouterbench",
        "mmr-bench",
        "acebench",
        "continuity-bench",
        "fusionfactory",
        "r2-router",
    }
    assert all(len(adapter.source_revision) == 40 for adapter in registry.adapters)
    assert all(adapter.track_ids for adapter in registry.adapters)
    assert all(adapter.limitations for adapter in registry.adapters)


def test_registry_matches_shared_research_inventory_pins_and_scope() -> None:
    registry = get_benchmark_registry()
    assert {adapter.id for adapter in registry.adapters} == set(
        RESEARCH_BENCHMARKS_BY_ADAPTER
    )
    for adapter in registry.adapters:
        inventory = RESEARCH_BENCHMARKS_BY_ADAPTER[adapter.id]
        assert adapter.source_revision == inventory["source_revision"]
        assert adapter.dataset_revision == inventory.get("dataset_revision")
        assert adapter.decision_unit == inventory["decision_unit"]
        assert adapter.action_space == inventory["action_space"]
        assert set(adapter.track_ids) == set(inventory["applicable_tracks"])


def test_research_inventory_is_loadable_as_packaged_resource() -> None:
    """The inventory must ship in a wheel, not depend on the Go source tree."""
    resource = files("cli.evaluation").joinpath(
        "golden/research_benchmark_inventory.v1.json"
    )
    document = json.loads(resource.read_text(encoding="utf-8"))
    assert document["schema_version"] == "evaluation-research-benchmark-inventory.v1"
    assert {item["adapter_id"] for item in document["benchmarks"]} == set(
        RESEARCH_BENCHMARKS_BY_ADAPTER
    )


def test_known_source_and_dataset_pins_are_not_mutable_labels() -> None:
    assert (
        get_benchmark_adapter("routerarena").source_revision
        == "fda4c53bcf9a979fd9c6f6bb6b713d6ab08ff43e"
    )
    xroute = get_benchmark_adapter("xroutebench")
    assert xroute.source_revision == "da3430baaea672743c3957457b0c76faba19876e"
    assert xroute.dataset_revision == "ea4b6e1b29d9a734f55f0a637baf326bad6aa681"


def test_source_verifier_requires_exact_clean_git_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sources" / "routerarena"
    source.mkdir(parents=True)
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "Evaluation Test")
    _git(source, "config", "user.email", "evaluation@example.invalid")
    (source / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "-q", "-m", "fixture")
    revision = _git(source, "rev-parse", "HEAD")

    descriptor = get_benchmark_adapter("routerarena")
    replacement = descriptor.model_copy(update={"source_revision": revision})
    monkeypatch.setattr(
        "cli.evaluation.benchmark_sources.get_benchmark_adapter",
        lambda _adapter_id: replacement,
    )

    receipt = verify_benchmark_source("routerarena", tmp_path / "sources")
    assert receipt.verified is True
    assert receipt.observed_source_revision == revision

    (source / "README.md").write_text("changed\n", encoding="utf-8")
    dirty = verify_benchmark_source("routerarena", tmp_path / "sources")
    assert dirty.verified is False
    assert dirty.source_clean is False

    _git(source, "checkout", "--", "README.md")
    (source / "poison.json").write_text("{}\n", encoding="utf-8")
    untracked = verify_benchmark_source("routerarena", tmp_path / "sources")
    assert untracked.verified is False
    assert untracked.source_clean is False


def test_source_verifier_rejects_symlinked_checkout(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    actual = tmp_path / "actual"
    actual.mkdir()
    root.mkdir()
    (root / "routerarena").symlink_to(actual, target_is_directory=True)
    with pytest.raises(SourceVerificationError, match="symlink"):
        verify_benchmark_source("routerarena", root)


def test_suite_manifest_rejects_unverified_or_label_collocated_input() -> None:
    receipt = BenchmarkSourceReceipt(
        adapter_id="routerarena",
        expected_source_revision="a" * 40,
        observed_source_revision="a" * 40,
        source_clean=True,
        verified=False,
    )
    artifacts = SuiteArtifactSet(
        visible_cases=_ref("a"),
        grading_cases=_ref("b"),
        license_manifest=_ref("c"),
    )
    subject = {
        "id": "routerarena-test",
        "name": "RouterArena test",
        "adapter_id": "routerarena",
        "source_receipt": receipt,
        "decision_unit": "query",
        "action_space": "one model",
        "track_ids": ("routing",),
        "split_protocol": "test",
        "case_count": 1,
        "arm_ids": (),
        "data_classification": "public",
        "redistribution": "metadata_only",
        "artifacts": artifacts,
        "limitations": ("fixture",),
    }
    qualification = BenchmarkQualificationReceipt(
        evidence_level="E0",
        manifest_subject_digest=qualification_manifest_subject_digest(subject),
        source_receipt_digest=digest_value(receipt),
        artifact_set_digest=digest_value(artifacts),
        executor_digest=NORMALIZED_REPLAY_EXECUTOR_DIGEST,
        qualification=UnqualifiedBenchmarkEvidence(
            origin="user_provided_import",
            parser_verified=False,
        ),
    )
    with pytest.raises(ValidationError, match="must be verified"):
        BenchmarkSuiteManifest(
            id="routerarena-test",
            name="RouterArena test",
            adapter_id="routerarena",
            source_receipt=receipt,
            revision="sha256:" + "d" * 64,
            decision_unit="query",
            action_space="one model",
            track_ids=("routing",),
            qualification_receipt=qualification,
            split_protocol="test",
            case_count=1,
            data_classification="public",
            redistribution="metadata_only",
            artifacts=artifacts,
            limitations=("fixture",),
        )

    assert SUITE_CONTRACT_VERSION == "evaluation-suite.v1"
    with pytest.raises(ValidationError, match="physically separate"):
        SuiteArtifactSet(
            visible_cases=_ref("a"),
            grading_cases=_ref("a"),
            license_manifest=_ref("c"),
        )
