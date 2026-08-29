from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cli.commands.eval import eval
from cli.evaluation.benchmark_registry import get_benchmark_adapter
from cli.evaluation.canonical import canonical_json_bytes
from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import BenchmarkSourceReceipt
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    LICENSE_CONTRACT_VERSION,
    BenchmarkSuiteInstallRequest,
    SuiteArtifactInstall,
    SuiteArtifactRole,
)
from click.testing import CliRunner


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")


def _write_bundle(root: Path) -> None:
    _write_jsonl(
        root / "visible/cases.jsonl",
        (
            CaseVisible(
                id="case-0",
                messages=({"role": "user", "content": "PRIVATE PROMPT"},),
            ).model_dump(mode="json"),
        ),
    )
    _write_jsonl(
        root / "grading/cases.jsonl",
        (
            CaseGrading(
                case_id="case-0",
                expected_route="private-route",
                preferred_arm_id="private-arm",
            ).model_dump(mode="json"),
        ),
    )
    license_path = root / "metadata/licenses.json"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": LICENSE_CONTRACT_VERSION,
                "licenses": [
                    {
                        "id": "upstream",
                        "name": "Fixture license",
                        "redistribution": "metadata_only",
                    }
                ],
            }
        )
    )


def _artifact(root: Path, role: SuiteArtifactRole) -> SuiteArtifactInstall:
    relative_path, media_type, _ = ARTIFACT_ROLE_LAYOUT[role]
    content = (root / relative_path).read_bytes()
    return SuiteArtifactInstall(
        role=role,
        relative_path=relative_path,
        digest="sha256:" + hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type=media_type,
    )


def _request(root: Path) -> BenchmarkSuiteInstallRequest:
    descriptor = get_benchmark_adapter("routerarena")
    receipt = BenchmarkSourceReceipt(
        adapter_id=descriptor.id,
        expected_source_revision=descriptor.source_revision,
        observed_source_revision=descriptor.source_revision,
        source_clean=True,
        verified=True,
    )
    return BenchmarkSuiteInstallRequest(
        id="routerarena-cli-test",
        name="RouterArena CLI test",
        adapter_id=descriptor.id,
        source_receipt=receipt,
        decision_unit=descriptor.decision_unit,
        action_space=descriptor.action_space,
        track_ids=("routing", "model_pool", "joint"),
        evidence_level_ceiling="E3",
        split_protocol="frozen CLI fixture split",
        case_count=1,
        arm_ids=("private-arm",),
        data_classification="restricted",
        redistribution="metadata_only",
        artifacts=tuple(
            _artifact(root, role)
            for role in ("visible_cases", "grading_cases", "license_manifest")
        ),
        limitations=("CLI fixture only",),
    )


def _write_request(path: Path, request: BenchmarkSuiteInstallRequest) -> None:
    path.write_text(
        json.dumps(request.model_dump(mode="json")),
        encoding="utf-8",
    )


def test_suite_install_list_and_show_keep_output_boundaries(
    tmp_path: Path, monkeypatch: Any
) -> None:
    runner = CliRunner()
    bundle = tmp_path / "bundle"
    request_path = tmp_path / "request.json"
    store_path = tmp_path / "suite-store"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    _write_bundle(bundle)
    request = _request(bundle)
    _write_request(request_path, request)
    monkeypatch.setattr(
        "cli.evaluation.suite_store.require_verified_benchmark_source",
        lambda _descriptor, _root: request.source_receipt,
    )

    installed_result = runner.invoke(
        eval,
        [
            "suite-install",
            "--request",
            str(request_path),
            "--bundle",
            str(bundle),
            "--source-root",
            str(source_root),
            "--suite-store",
            str(store_path),
        ],
    )
    assert installed_result.exit_code == 0, installed_result.output
    installed = json.loads(installed_result.output)
    assert installed["id"] == "routerarena-cli-test"
    assert installed["revision"].startswith("sha256:")
    assert set(installed["artifacts"]) >= {
        "visible_cases",
        "grading_cases",
        "license_manifest",
    }

    listed_result = runner.invoke(
        eval, ["suite-list", "--suite-store", str(store_path)]
    )
    assert listed_result.exit_code == 0, listed_result.output
    listed = json.loads(listed_result.output)
    assert listed["schema_version"] == SCHEMA_VERSION
    assert [suite["id"] for suite in listed["suites"]] == ["routerarena-cli-test"]
    encoded_list = listed_result.output
    assert "artifacts" not in encoded_list
    assert "source_receipt" not in encoded_list
    assert "private-arm" not in encoded_list
    assert "private-route" not in encoded_list
    assert "PRIVATE PROMPT" not in encoded_list

    shown_result = runner.invoke(
        eval,
        [
            "suite-show",
            "routerarena-cli-test",
            "--suite-store",
            str(store_path),
        ],
    )
    assert shown_result.exit_code == 0, shown_result.output
    assert json.loads(shown_result.output) == installed
    assert "PRIVATE PROMPT" not in shown_result.output
    assert "private-route" not in shown_result.output


def test_suite_install_strictly_rejects_unknown_request_fields(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    bundle = tmp_path / "bundle"
    request_path = tmp_path / "request.json"
    store_path = tmp_path / "suite-store"
    source_root = tmp_path / "sources"
    source_root.mkdir()
    _write_bundle(bundle)
    payload = _request(bundle).model_dump(mode="json")
    payload["executable_adapter"] = "do-not-run.py"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        eval,
        [
            "suite-install",
            "--request",
            str(request_path),
            "--bundle",
            str(bundle),
            "--source-root",
            str(source_root),
            "--suite-store",
            str(store_path),
        ],
    )

    assert result.exit_code != 0
    assert "executable_adapter" in result.output
    assert "extra" in result.output.lower()
    assert not store_path.exists()


def test_suite_list_uses_private_default_store(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(eval, ["suite-list"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["suites"] == []
    default_store = tmp_path / ".vllm-sr/evaluation-suites"
    assert default_store.is_dir()
    assert default_store.stat().st_mode & 0o777 == 0o700


def test_suite_show_reports_missing_suite_as_user_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        eval,
        [
            "suite-show",
            "missing-suite",
            "--suite-store",
            str(tmp_path / "suite-store"),
        ],
    )

    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "missing" in result.output.lower()
