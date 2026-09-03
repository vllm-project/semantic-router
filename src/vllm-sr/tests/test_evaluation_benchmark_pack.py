from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from cli.commands.eval import eval
from cli.evaluation import suite_store_install
from cli.evaluation.canonical import canonical_json_bytes
from cli.evaluation.constants import SCHEMA_VERSION, TRACK_IDS
from cli.evaluation.contracts import CaseGrading, CaseVisible, VisibleCaseSet
from cli.evaluation.errors import SuiteStoreError
from cli.evaluation.evidence import ExecutionRecord, RoutingDiagnostic
from cli.evaluation.evidence_source_ids import LIVE_ROUTING_EVIDENCE_SOURCE_ID
from cli.evaluation.execution_contract import NORMALIZED_LIVE_EXECUTOR_ID
from cli.evaluation.live_executor import LiveRawResult
from cli.evaluation.normalized_suite_live_executor import (
    execute_normalized_suite_live,
)
from cli.evaluation.orchestrator import run_evaluation
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.suite_contract import SUITE_CONTRACT_VERSION, BenchmarkSourceReceipt
from cli.evaluation.suite_install_contract import LICENSE_CONTRACT_VERSION
from cli.evaluation.suite_store import NormalizedSuiteStore
from click.testing import CliRunner
from evaluation_normalized_suite_test_support import (
    _live_manifest,
    _manifest,
    _target_mixture,
)


def _write_jsonl(path: Path, rows: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _write_pack(
    root: Path,
    *,
    track_ids: tuple[str, ...] = ("routing",),
    image: bool = False,
    expected_route: str | None = "arm-strong",
    expected_answer: str | None = None,
    extra_manifest: dict[str, object] | None = None,
) -> None:
    bundle = root / "bundle"
    image_data = b"\x00"
    content: object = "private pack prompt"
    if image:
        content = (
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,AA==",
                    "detail": "low",
                },
            },
        )
    _write_jsonl(
        bundle / "visible/cases.jsonl",
        (
            CaseVisible(
                id="case-1",
                track_ids=track_ids,
                messages=({"role": "user", "content": content},),
                modality="image" if image else "text",
            ).model_dump(mode="json"),
        ),
    )
    _write_jsonl(
        bundle / "grading/cases.jsonl",
        (
            CaseGrading(
                case_id="case-1",
                expected_route=expected_route,
                expected_answer=expected_answer,
            ).model_dump(mode="json"),
        ),
    )
    if image:
        _write_jsonl(
            bundle / "metadata/media.jsonl",
            (
                {
                    "schema_version": SUITE_CONTRACT_VERSION,
                    "id": "image-1",
                    "digest": "sha256:" + hashlib.sha256(image_data).hexdigest(),
                    "media_type": "image/png",
                    "size_bytes": len(image_data),
                    "modality": "image",
                    "license_id": "pack",
                },
            ),
        )
    license_path = bundle / "metadata/licenses.json"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": LICENSE_CONTRACT_VERSION,
                "licenses": [
                    {
                        "id": "pack",
                        "name": "Example pack license",
                        "redistribution": "metadata_only",
                    }
                ],
            }
        )
    )
    manifest: dict[str, object] = {
        "schema_version": "evaluation-benchmark-pack.v1",
        "id": "acme-routing-v1",
        "benchmark_id": "acme.routing",
        "name": "Acme routing benchmark",
        "decision_unit": "request",
        "action_space": "one model",
        "track_ids": list(track_ids),
        "split_protocol": "fixed public test split",
        "arm_ids": ["arm-fast", "arm-strong"],
        "data_classification": "restricted",
        "redistribution": "metadata_only",
        "limitations": ["Example fixture only."],
    }
    manifest.update(extra_manifest or {})
    (root / "benchmark.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def _git(root: Path, *args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=capture,
        text=True,
    )
    return result.stdout.strip() if capture else ""


def _commit_pack(root: Path) -> str:
    commands = (
        ("init", "--quiet"),
        ("config", "user.name", "Benchmark Pack Test"),
        ("config", "user.email", "benchmark-pack@example.invalid"),
        ("add", "."),
        ("commit", "--quiet", "-m", "benchmark pack fixture"),
    )
    for command in commands:
        _git(root, *command)
    return _git(root, "rev-parse", "HEAD", capture=True)


def test_benchmark_install_accepts_data_only_pack_and_projects_catalog(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(pack)
    revision = _commit_pack(pack)
    store = tmp_path / "suite-store"
    runner = CliRunner()

    installed = runner.invoke(
        eval,
        ["benchmark-install", "--pack", str(pack), "--suite-store", str(store)],
    )

    assert installed.exit_code == 0, installed.output
    manifest = json.loads(installed.output)
    assert manifest["adapter_id"] == "acme.routing"
    assert manifest["source_receipt"] == {
        "adapter_id": "acme.routing",
        "dataset_clean": None,
        "expected_dataset_revision": None,
        "expected_source_revision": revision,
        "observed_dataset_revision": None,
        "observed_source_revision": revision,
        "schema_version": "benchmark-source.v1",
        "source_clean": True,
        "source_kind": "benchmark_pack",
        "verified": True,
    }

    catalog_result = runner.invoke(
        eval,
        ["catalog", "--suite-store", str(store)],
    )
    assert catalog_result.exit_code == 0, catalog_result.output
    catalog = json.loads(catalog_result.output)
    suite = next(item for item in catalog["suites"] if item["id"] == manifest["id"])
    assert suite["modes"] == ["replay", "live"]
    assert suite["executors"] == {
        "live": "normalized-suite-live.v1",
        "replay": "normalized-suite-replay.v1",
    }
    assert suite["evidence_level"] == "E0"
    methods = {method["id"]: method for method in suite["methods"]}
    assert methods["normalized.acme.routing.routing.v1"] == {
        "id": "normalized.acme.routing.routing.v1",
        "track_id": "routing",
        "qualified_gate_ids": [],
        "evidence_source": "normalized_import",
        "status": "configured",
        "reason": None,
    }
    assert methods["benchmark-pack.server-live.routing.v1"] == {
        "id": "benchmark-pack.server-live.routing.v1",
        "track_id": "routing",
        "qualified_gate_ids": [],
        "evidence_source": "live_runtime",
        "status": "configured",
        "reason": None,
    }
    assert catalog["schema_version"] == SCHEMA_VERSION
    assert "private pack prompt" not in catalog_result.output

    suite_store = NormalizedSuiteStore(store)
    replay = _manifest("benchmark-pack-replay", (manifest["id"],), suite_store)
    replay = replay.with_semantic_updates(track_ids=("routing",), sample_limit=1)
    report = run_evaluation(
        replay,
        LocalArtifactStore(tmp_path / "evaluation-store"),
        suite_store=suite_store,
    )
    assert report.run.evidence_level == "E0"
    assert report.tracks[0].track_id == "routing"


def test_benchmark_pack_without_a_hidden_route_stays_replay_only(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(pack, expected_route=None)
    _commit_pack(pack)
    store = tmp_path / "suite-store"
    runner = CliRunner()

    installed = runner.invoke(
        eval,
        ["benchmark-install", "--pack", str(pack), "--suite-store", str(store)],
    )
    assert installed.exit_code == 0, installed.output

    catalog_result = runner.invoke(
        eval,
        ["catalog", "--suite-store", str(store)],
    )
    assert catalog_result.exit_code == 0, catalog_result.output
    suite_id = json.loads(installed.output)["id"]
    suite = next(
        item
        for item in json.loads(catalog_result.output)["suites"]
        if item["id"] == suite_id
    )
    assert suite["modes"] == ["replay"]
    assert [method["id"] for method in suite["methods"]] == [
        "normalized.acme.routing.routing.v1"
    ]


def test_benchmark_pack_without_a_hidden_multimodal_answer_stays_replay_only(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(
        pack,
        track_ids=("multimodal",),
        image=True,
        expected_route=None,
        expected_answer=None,
    )
    _commit_pack(pack)
    store = tmp_path / "suite-store"
    manifest = NormalizedSuiteStore(store).install_pack(pack)

    catalog_result = CliRunner().invoke(
        eval,
        ["catalog", "--suite-store", str(store)],
    )

    assert catalog_result.exit_code == 0, catalog_result.output
    suite = next(
        item
        for item in json.loads(catalog_result.output)["suites"]
        if item["id"] == manifest.id
    )
    assert suite["modes"] == ["replay"]
    assert [method["id"] for method in suite["methods"]] == [
        "normalized.acme.routing.multimodal.v1"
    ]


def test_benchmark_pack_admits_only_complete_platform_live_tracks(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(
        pack,
        track_ids=TRACK_IDS,
        image=True,
        expected_answer="answer",
    )
    _commit_pack(pack)
    store = tmp_path / "suite-store"
    manifest = NormalizedSuiteStore(store).install_pack(pack)

    catalog_result = CliRunner().invoke(
        eval,
        ["catalog", "--suite-store", str(store)],
    )
    assert catalog_result.exit_code == 0, catalog_result.output
    suite = next(
        item
        for item in json.loads(catalog_result.output)["suites"]
        if item["id"] == manifest.id
    )
    live_tracks = {
        method["track_id"]
        for method in suite["methods"]
        if method["id"].startswith("benchmark-pack.server-live.")
    }
    assert live_tracks == {"routing", "model_pool", "joint", "multimodal", "capacity"}
    assert live_tracks.isdisjoint({"agentic", "preference", "safety"})


def test_benchmark_pack_receipt_cannot_claim_a_dirty_checkout_is_verified() -> None:
    with pytest.raises(ValueError, match="verified benchmark source must be clean"):
        BenchmarkSourceReceipt(
            source_kind="benchmark_pack",
            adapter_id="acme.routing",
            expected_source_revision="a" * 40,
            observed_source_revision="a" * 40,
            source_clean=False,
            verified=True,
        )


def test_benchmark_install_rejects_dirty_or_executable_packs(tmp_path: Path) -> None:
    dirty_pack = tmp_path / "dirty-pack"
    dirty_pack.mkdir()
    _write_pack(dirty_pack)
    _commit_pack(dirty_pack)
    (dirty_pack / "benchmark.yaml").write_text(
        (dirty_pack / "benchmark.yaml").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    dirty = CliRunner().invoke(
        eval,
        [
            "benchmark-install",
            "--pack",
            str(dirty_pack),
            "--suite-store",
            str(tmp_path / "dirty-store"),
        ],
    )
    assert dirty.exit_code != 0
    assert "clean Git checkout" in dirty.output

    executable_pack = tmp_path / "executable-pack"
    executable_pack.mkdir()
    _write_pack(executable_pack, extra_manifest={"runner": "python benchmark.py"})
    _commit_pack(executable_pack)
    executable = CliRunner().invoke(
        eval,
        [
            "benchmark-install",
            "--pack",
            str(executable_pack),
            "--suite-store",
            str(tmp_path / "executable-store"),
        ],
    )
    assert executable.exit_code != 0
    assert "does not match its contract" in executable.output


@pytest.mark.parametrize(
    ("manifest_suffix", "expected_error"),
    (
        ("\nid: duplicate\n", "duplicate key"),
        ("\nanchor_probe: &probe value\n", "aliases are not supported"),
    ),
)
def test_benchmark_install_rejects_ambiguous_yaml(
    tmp_path: Path,
    manifest_suffix: str,
    expected_error: str,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(pack)
    manifest_path = pack / "benchmark.yaml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + manifest_suffix,
        encoding="utf-8",
    )
    _commit_pack(pack)

    result = CliRunner().invoke(
        eval,
        [
            "benchmark-install",
            "--pack",
            str(pack),
            "--suite-store",
            str(tmp_path / "suite-store"),
        ],
    )

    assert result.exit_code != 0
    assert expected_error in result.output


def test_benchmark_install_does_not_publish_invalid_manifest_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(pack)
    _commit_pack(pack)
    store = NormalizedSuiteStore(tmp_path / "suite-store")
    original_stage = suite_store_install.stage_suite_artifacts
    staged = False

    def stage_then_invalidate_manifest(*args: object, **kwargs: object):
        nonlocal staged
        artifacts = original_stage(*args, **kwargs)
        staged = True
        manifest_path = pack / "benchmark.yaml"
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8") + "unexpected: true\n",
            encoding="utf-8",
        )
        return artifacts

    monkeypatch.setattr(
        suite_store_install,
        "stage_suite_artifacts",
        stage_then_invalidate_manifest,
    )

    with pytest.raises(SuiteStoreError, match="does not match its contract"):
        store.install_pack(pack)

    assert staged
    assert store.list_suite_manifests() == ()
    assert tuple(store.index.iterdir()) == ()
    assert tuple(store.manifests.iterdir()) == ()
    assert tuple(store.objects.rglob(".install-*")) == ()


def test_benchmark_install_rejects_unknown_bundle_files(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(pack)
    (pack / "bundle/run.py").write_text(
        "raise SystemExit('never run')\n", encoding="utf-8"
    )
    _commit_pack(pack)

    result = CliRunner().invoke(
        eval,
        [
            "benchmark-install",
            "--pack",
            str(pack),
            "--suite-store",
            str(tmp_path / "suite-store"),
        ],
    )

    assert result.exit_code != 0
    assert "unknown file 'run.py'" in result.output


def test_benchmark_install_rejects_ignored_pack_data(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(pack)
    (pack / ".gitignore").write_text("benchmark.yaml\nbundle/\n", encoding="utf-8")
    _git(pack, "init", "--quiet")
    _git(pack, "config", "user.name", "Benchmark Pack Test")
    _git(pack, "config", "user.email", "benchmark-pack@example.invalid")
    _git(pack, "add", ".gitignore")
    _git(pack, "commit", "--quiet", "-m", "ignore fixture")

    result = CliRunner().invoke(
        eval,
        [
            "benchmark-install",
            "--pack",
            str(pack),
            "--suite-store",
            str(tmp_path / "suite-store"),
        ],
    )

    assert result.exit_code != 0
    assert "clean Git checkout" in result.output


def test_benchmark_pack_runs_live_against_a_frozen_mixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    _write_pack(pack)
    _commit_pack(pack)
    store = NormalizedSuiteStore(tmp_path / "suite-store")
    suite_id = store.install_pack(pack).id
    manifest = _live_manifest(
        "benchmark-pack-live-routing",
        suite_id,
        store,
        track_ids=("routing",),
    )

    def execute(visible: VisibleCaseSet, **kwargs: object) -> LiveRawResult:
        case = visible.cases[0]
        assert kwargs["track_ids"] == ("routing",)
        assert kwargs["mixture"] == _target_mixture()
        return LiveRawResult(
            records=[
                ExecutionRecord(
                    id="routing-case-1",
                    track_id="routing",
                    case_id=case.id,
                    attempt_id="attempt-case-1",
                    status="succeeded",
                    selected_arm_id="arm-strong",
                    selection_status="selected",
                    selection_method="static",
                    recipe="target-recipe",
                    trace_digest="sha256:" + "a" * 64,
                    success=True,
                    fallback=False,
                    latency_ms=2.0,
                    evidence_kind=LIVE_ROUTING_EVIDENCE_SOURCE_ID,
                    broker_receipt="sha256:" + "b" * 64,
                )
            ],
            discovered_entrypoints=("entrypoint-a",),
            routing_traces=(
                RoutingDiagnostic(
                    case_id=case.id,
                    selected_model="provider-strong",
                    selection_status="selected",
                ),
            ),
            chat_results={},
            model_pool_results={},
            model_pool_arm_ids=(),
            joint_results={},
        )

    monkeypatch.setattr(
        "cli.evaluation.normalized_suite_live_executor.execute_live_raw",
        execute,
    )
    result = execute_normalized_suite_live(
        manifest=manifest,
        store=store,
        manifests=(store.get_suite_manifest(suite_id),),
        executor_id=NORMALIZED_LIVE_EXECUTOR_ID,
    )

    assert len(result.records) == 1
    assert result.records[0].quality == 1.0
    assert result.records[0].grader == "normalized-suite-hidden-route-label.v1"
    assert result.records[0].evidence_kind == LIVE_ROUTING_EVIDENCE_SOURCE_ID
