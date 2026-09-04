from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from cli.evaluation.benchmark_registry import get_benchmark_adapter
from cli.evaluation.benchmark_sources import SourceVerificationError
from cli.evaluation.canonical import canonical_json_bytes
from cli.evaluation.contracts import CaseGrading, CaseVisible
from cli.evaluation.suite_contract import BenchmarkSourceReceipt, NormalizedOutcome
from cli.evaluation.suite_install_contract import (
    ARTIFACT_ROLE_LAYOUT,
    LICENSE_CONTRACT_VERSION,
    BenchmarkSuiteInstallRequest,
    SuiteArtifactInstall,
)
from cli.evaluation.suite_store import NormalizedSuiteStore, SuiteStoreError
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _trusted_source_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def verified(descriptor: Any, _source_root: Path) -> BenchmarkSourceReceipt:
        return BenchmarkSourceReceipt(
            adapter_id=descriptor.id,
            expected_source_revision=descriptor.source_revision,
            observed_source_revision=descriptor.source_revision,
            expected_dataset_revision=descriptor.dataset_revision,
            observed_dataset_revision=descriptor.dataset_revision,
            source_clean=True,
            dataset_clean=(True if descriptor.dataset_revision else None),
            verified=True,
        )

    monkeypatch.setattr(
        "cli.evaluation.suite_store.require_verified_benchmark_source", verified
    )


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")


def _write_license(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
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


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _artifact(root: Path, role: str) -> SuiteArtifactInstall:
    relative_path, media_type, _ = ARTIFACT_ROLE_LAYOUT[role]  # type: ignore[index]
    digest, size = _file_identity(root / relative_path)
    return SuiteArtifactInstall(
        role=role,
        relative_path=relative_path,
        digest=digest,
        size_bytes=size,
        media_type=media_type,
    )


def _receipt(*, clean: bool = True) -> BenchmarkSourceReceipt:
    descriptor = get_benchmark_adapter("routerarena")
    return BenchmarkSourceReceipt(
        adapter_id=descriptor.id,
        expected_source_revision=descriptor.source_revision,
        observed_source_revision=descriptor.source_revision,
        source_clean=clean,
        verified=clean,
    )


def _visible(case_id: str, text: str = "TOP SECRET PROMPT") -> dict[str, Any]:
    return CaseVisible(
        id=case_id,
        messages=({"role": "user", "content": text},),
        tags=("private-input",),
    ).model_dump(mode="json")


def _grading(case_id: str) -> dict[str, Any]:
    return CaseGrading(
        case_id=case_id,
        expected_route="private-route",
        preferred_arm_id="private-arm",
    ).model_dump(mode="json")


def _bundle(root: Path, *, count: int = 2, prompt: str = "TOP SECRET PROMPT") -> None:
    _write_jsonl(
        root / "visible/cases.jsonl",
        (_visible(f"case-{index}", prompt) for index in range(count)),
    )
    _write_jsonl(
        root / "grading/cases.jsonl",
        (_grading(f"case-{index}") for index in range(count)),
    )
    _write_license(root / "metadata/licenses.json")


def _request(
    root: Path,
    *,
    suite_id: str = "routerarena-normalized-test",
    name: str = "RouterArena normalized test",
    count: int = 2,
    receipt: BenchmarkSourceReceipt | None = None,
) -> BenchmarkSuiteInstallRequest:
    descriptor = get_benchmark_adapter("routerarena")
    return BenchmarkSuiteInstallRequest(
        id=suite_id,
        name=name,
        adapter_id=descriptor.id,
        source_receipt=receipt or _receipt(),
        decision_unit=descriptor.decision_unit,
        action_space=descriptor.action_space,
        track_ids=("routing", "model_pool", "joint"),
        evidence_level_ceiling="E3",
        split_protocol="frozen fixture split",
        case_count=count,
        arm_ids=("private-arm",),
        data_classification="restricted",
        redistribution="metadata_only",
        artifacts=tuple(
            _artifact(root, role)
            for role in ("visible_cases", "grading_cases", "license_manifest")
        ),
        limitations=("fixture only",),
    )


def test_install_is_idempotent_immutable_and_physically_separated(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    store = NormalizedSuiteStore(tmp_path / "store")
    request = _request(bundle)

    first = store.install(request, bundle, source_root=bundle.parent)
    second = store.install(request, bundle, source_root=bundle.parent)

    assert first == second
    assert first.revision.startswith("sha256:")
    assert first.artifacts.visible_cases.digest != first.artifacts.grading_cases.digest
    assert (
        store.objects
        / "visible"
        / "sha256"
        / first.artifacts.visible_cases.digest.removeprefix("sha256:")
    ).is_file()
    assert (
        store.objects
        / "grading"
        / "sha256"
        / first.artifacts.grading_cases.digest.removeprefix("sha256:")
    ).is_file()
    assert len(list(store.manifests.iterdir())) == 1
    manifest_path = next(store.manifests.iterdir())
    manifest_digest, _ = _file_identity(manifest_path)
    assert manifest_path.name == manifest_digest.removeprefix("sha256:")

    visible = list(store.load_jsonl(first.id, "visible_cases"))
    grading = list(store.load_jsonl(first.id, "grading_cases"))
    assert [case.id for case in visible] == ["case-0", "case-1"]
    assert [case.case_id for case in grading] == ["case-0", "case-1"]

    changed = _request(bundle, name="same ID, changed metadata")
    with pytest.raises(SuiteStoreError, match="immutable"):
        store.install(changed, bundle, source_root=bundle.parent)


def test_catalog_never_contains_private_records_refs_or_arm_ids(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    _write_jsonl(
        bundle / "grading/outcomes.jsonl",
        (
            NormalizedOutcome(
                case_id="case-0",
                arm_id="private-arm",
                quality=0.75,
                grader_id="private-outcome-grader",
                split="fixture",
                source_record_digest="sha256:" + "a" * 64,
            ),
        ),
    )
    store = NormalizedSuiteStore(tmp_path / "store")
    request = _request(bundle)
    request = request.model_copy(
        update={"artifacts": (*request.artifacts, _artifact(bundle, "outcomes"))}
    )
    manifest = store.install(request, bundle, source_root=bundle.parent)

    catalog = store.get_catalog_suite(manifest.id)
    listed = store.list_catalog_suites()
    encoded = json.dumps(catalog.model_dump(mode="json"), sort_keys=True)

    assert listed == (catalog,)
    assert "TOP SECRET PROMPT" not in encoded
    assert "private-route" not in encoded
    assert "private-arm" not in encoded
    assert "private-outcome-grader" not in encoded
    assert "artifacts" not in encoded
    assert first_digest_absent(manifest.artifacts.visible_cases.digest, encoded)
    manifest_blob_digest = "sha256:" + next(store.manifests.iterdir()).name
    assert manifest_blob_digest not in encoded
    assert catalog.modes == ("replay",)
    outcomes = list(store.load_jsonl(manifest.id, "outcomes"))
    assert len(outcomes) == 1
    assert outcomes[0].grader_id == "private-outcome-grader"


def first_digest_absent(digest: str, encoded: str) -> bool:
    """Make the sensitive-catalog assertion readable without exposing internals."""

    return digest not in encoded


def test_install_ignores_a_forged_caller_receipt(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    store = NormalizedSuiteStore(tmp_path / "store")
    forged = _receipt().model_copy(update={"observed_source_revision": "0" * 40})

    installed = store.install(
        _request(bundle, receipt=forged), bundle, source_root=bundle.parent
    )

    assert installed.source_receipt.observed_source_revision != "0" * 40


def test_install_rejects_failed_system_source_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)

    def reject(_descriptor: Any, _root: Path) -> BenchmarkSourceReceipt:
        raise SourceVerificationError("source attestation failed")

    monkeypatch.setattr(
        "cli.evaluation.suite_store.require_verified_benchmark_source", reject
    )
    with pytest.raises(SuiteStoreError, match="source attestation failed"):
        NormalizedSuiteStore(tmp_path / "store").install(
            _request(bundle), bundle, source_root=bundle.parent
        )


def test_closed_artifact_contract_rejects_paths_media_types_duplicates_and_extra_fields(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    valid = _artifact(bundle, "visible_cases").model_dump(mode="json")

    with pytest.raises(ValidationError, match="requires relative path"):
        SuiteArtifactInstall.model_validate(
            {**valid, "relative_path": "../../labels.jsonl"}
        )
    with pytest.raises(ValidationError, match="requires media type"):
        SuiteArtifactInstall.model_validate({**valid, "media_type": "text/plain"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        SuiteArtifactInstall.model_validate({**valid, "executable": "adapter.py"})

    request = _request(bundle)
    with pytest.raises(ValidationError, match="roles must be unique"):
        BenchmarkSuiteInstallRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "artifacts": [
                    *request.model_dump(mode="json")["artifacts"],
                    valid,
                ],
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        BenchmarkSuiteInstallRequest.model_validate(
            {**request.model_dump(mode="json"), "browser_prompt": "leak"}
        )

    bypassed = request.model_copy(
        update={
            "artifacts": (
                request.artifacts[0].model_copy(
                    update={"relative_path": "visible/other.jsonl"}
                ),
                *request.artifacts[1:],
            )
        }
    )
    with pytest.raises(SuiteStoreError, match="invalid suite install request"):
        NormalizedSuiteStore(tmp_path / "store").install(
            bypassed, bundle, source_root=bundle.parent
        )


@pytest.mark.parametrize("field", ["digest", "size_bytes"])
def test_declared_artifact_identity_is_strict(tmp_path: Path, field: str) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    request = _request(bundle)
    visible = request.artifacts[0]
    replacement = "sha256:" + "0" * 64 if field == "digest" else visible.size_bytes + 1
    artifacts = (
        visible.model_copy(update={field: replacement}),
        *request.artifacts[1:],
    )
    changed = request.model_copy(update={"artifacts": artifacts})

    with pytest.raises(SuiteStoreError, match=r"declared size|digest and size"):
        NormalizedSuiteStore(tmp_path / "store").install(
            changed, bundle, source_root=bundle.parent
        )


def test_bundle_symlinks_and_symlinked_parents_are_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    request = _request(bundle)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    (bundle / "visible/cases.jsonl").unlink()
    (bundle / "visible/cases.jsonl").symlink_to(outside)

    with pytest.raises(SuiteStoreError, match=r"symlink|escapes"):
        NormalizedSuiteStore(tmp_path / "store-a").install(
            request, bundle, source_root=bundle.parent
        )

    bundle_b = tmp_path / "bundle-b"
    _bundle(bundle_b)
    request_b = _request(bundle_b, suite_id="second-suite")
    actual_grading = tmp_path / "actual-grading"
    (bundle_b / "grading").rename(actual_grading)
    (bundle_b / "grading").symlink_to(actual_grading, target_is_directory=True)
    with pytest.raises(SuiteStoreError, match=r"parent.*symlink"):
        NormalizedSuiteStore(tmp_path / "store-b").install(
            request_b, bundle_b, source_root=bundle_b.parent
        )


def test_store_enforces_private_permissions_and_rejects_permission_drift(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    store = NormalizedSuiteStore(tmp_path / "store")
    manifest = store.install(_request(bundle), bundle, source_root=bundle.parent)

    owned = [store.root, *store.root.rglob("*")]
    directories = [path for path in owned if path.is_dir()]
    files = [path for path in owned if path.is_file()]
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)

    visible_path = (
        store.objects
        / "visible"
        / "sha256"
        / manifest.artifacts.visible_cases.digest.removeprefix("sha256:")
    )
    visible_path.chmod(0o644)
    with pytest.raises(SuiteStoreError, match="mode 0600"):
        list(store.load_jsonl(manifest.id, "visible_cases"))

    unsafe_root = tmp_path / "unsafe-store"
    unsafe_root.mkdir(mode=0o755)
    with pytest.raises(SuiteStoreError, match="mode 0700"):
        NormalizedSuiteStore(unsafe_root)


def test_large_artifacts_install_and_load_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "large-bundle"
    count = 1536
    _bundle(bundle, count=count, prompt="x" * 4096)
    request = _request(bundle, count=count)
    assert (bundle / "visible/cases.jsonl").stat().st_size > 6 * 1024 * 1024

    def forbidden_read_bytes(_path: Path) -> bytes:
        raise AssertionError("large artifacts must never use Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    store = NormalizedSuiteStore(tmp_path / "store")
    manifest = store.install(request, bundle, source_root=bundle.parent)
    assert sum(1 for _ in store.load_jsonl(manifest.id, "visible_cases")) == count


def test_normalized_record_schema_is_strict_at_install_time(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle, count=1)
    visible_path = bundle / "visible/cases.jsonl"
    invalid = _visible("case-0")
    invalid["expected_route"] = "label-leak"
    _write_jsonl(visible_path, (invalid,))
    request = _request(bundle, count=1)

    with pytest.raises(SuiteStoreError, match="invalid normalized visible_cases"):
        NormalizedSuiteStore(tmp_path / "store").install(
            request, bundle, source_root=bundle.parent
        )


def test_object_content_drift_is_detected_before_reuse(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    store = NormalizedSuiteStore(tmp_path / "store")
    request = _request(bundle)
    manifest = store.install(request, bundle, source_root=bundle.parent)
    object_path = (
        store.objects
        / "visible"
        / "sha256"
        / manifest.artifacts.visible_cases.digest.removeprefix("sha256:")
    )
    object_path.write_bytes(b"drift\n")
    object_path.chmod(0o600)

    with pytest.raises(SuiteStoreError, match="corrupt"):
        store.install(request, bundle, source_root=bundle.parent)
