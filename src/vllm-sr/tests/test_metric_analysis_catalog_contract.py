from __future__ import annotations

import ast
import copy
import json
import re
import shutil
import zipfile
from pathlib import Path

import pytest
from cli.evaluation import metric_analysis_catalog as catalog


def test_catalog_is_complete_sorted_and_fail_closed() -> None:
    assert len(catalog.STATIC_METRIC_IDS) == 136
    assert tuple(sorted(catalog.STATIC_METRIC_IDS)) == catalog.STATIC_METRIC_IDS
    assert catalog.DYNAMIC_FAMILY_IDS == (
        "capacity-level",
        "model-pool-arm",
        "multimodal-modality",
        "routing-recipe-e1-input",
        "routing-recipe-e2-projection",
        "routing-recipe-e2-recall",
    )
    for metric_id in catalog.STATIC_METRIC_IDS:
        match = catalog.resolve_metric_analysis(metric_id)
        assert match.metric_id == metric_id
        assert match.family_id is None

    assert (
        catalog.resolve_metric_analysis("routing.accuracy").specification.analysis_ref
        == "routing.case.ratio"
    )
    recovery = catalog.resolve_metric_analysis(
        "agentic.recovery_cluster_pass_rate_lower_95"
    )
    assert recovery.specification.analysis_ref == (
        "agentic.recovery.cluster-wilson-bounds"
    )
    assert recovery.specification.analysis_unit == "recovery_cluster"
    assert recovery.specification.cluster_unit == "cluster_id"
    assert recovery.specification.weighting == "uniform_cluster"
    arm = catalog.resolve_metric_analysis("model_pool.arm.fast.quality")
    assert arm.family_id == "model-pool-arm"
    assert dict(arm.captures) == {"arm_id": "fast", "statistic": "quality"}
    assert arm.specification.planned_unit_projection["source"] == (
        "frozen_model_pool_matrix"
    )
    capacity = catalog.resolve_metric_analysis("capacity.level.16.latency_p95_ms")
    assert capacity.specification.analysis_unit == "measurement_request"
    assert capacity.specification.weighting == "uniform_request"
    cluster_bound = catalog.resolve_metric_analysis(
        "capacity.level.16.error_rate_upper_bound"
    )
    assert cluster_bound.specification.analysis_unit == "measurement_cluster"
    assert cluster_bound.specification.cluster_unit == "measurement_cluster"
    assert cluster_bound.specification.weighting == "worst_cluster"
    for metric_id in (
        "capacity.error_rate",
        "capacity.success_rate",
        "capacity.level.16.error_rate",
        "capacity.level.16.success_rate",
    ):
        rate = catalog.resolve_metric_analysis(metric_id)
        assert rate.specification.analysis_unit == "measurement_cluster"
        assert rate.specification.weighting == "uniform_cluster"
        assert rate.specification.analysis_ref.endswith("repetition-rate-mean")
    recipe = catalog.resolve_metric_analysis(
        "routing_recipe.e1.signal.u-c2lnbmFsLnRvcGljLnYx.present_rate"
    )
    assert catalog.decode_metric_subject_id(recipe.captures["subject_id"]) == (
        "signal.topic.v1"
    )

    for unknown in (
        "routing.injected",
        "model_pool.arm.model.v1.quality",
        "model_pool.arm.u-abc.quality",
        "capacity.level.0.success_rate",
        "capacity.level.16.injected",
        "routing_recipe.e2.feasible_oracle_recall_at_65",
    ):
        with pytest.raises(ValueError, match=r"unknown|canonical|base64url|range"):
            catalog.resolve_metric_analysis(unknown)


def test_portable_subject_encoding_golden_vectors() -> None:
    document = json.loads(catalog.metric_analysis_catalog_bytes())
    for vector in document["identifier_encoding"]["vectors"]:
        assert catalog.encode_metric_subject_id(vector["raw"]) == vector["encoded"]
        assert catalog.decode_metric_subject_id(vector["encoded"]) == vector["raw"]

    assert catalog.encode_metric_subject_id("domain:reasoning") == (
        "u-ZG9tYWluOnJlYXNvbmluZw"
    )
    assert catalog.encode_metric_subject_id("classifier:risk:RISKY") == (
        "u-Y2xhc3NpZmllcjpyaXNrOlJJU0tZ"
    )
    assert "." not in catalog.encode_metric_subject_id("signal.topic.v1")
    with pytest.raises(ValueError, match="portable"):
        catalog.encode_metric_subject_id("not portable")
    with pytest.raises(ValueError, match="canonical"):
        catalog.decode_metric_subject_id("u-")


def test_overlapping_dynamic_family_document_is_rejected() -> None:
    document = copy.deepcopy(catalog._DOCUMENT)
    duplicate = copy.deepcopy(document["dynamic_families"][0])
    duplicate["id"] = "capacity-level-shadow"
    document["dynamic_families"].append(duplicate)
    document["dynamic_families"].sort(key=lambda item: item["id"])
    with pytest.raises(RuntimeError, match="overlap"):
        catalog._validate_document(document)


def test_catalog_validator_accepts_a_sorted_referenced_extension() -> None:
    document = copy.deepcopy(catalog._DOCUMENT)
    template = copy.deepcopy(
        next(
            item
            for item in document["analysis_templates"]
            if item["id"] == "routing.case.ratio"
        )
    )
    template["id"] = "routing.catalog-extensibility-probe"
    document["analysis_templates"].append(template)
    document["analysis_templates"].sort(key=lambda item: item["id"])
    document["static_metrics"].append(
        {
            "id": "routing.catalog_extensibility_probe",
            "analysis_ref": template["id"],
        }
    )
    document["static_metrics"].sort(key=lambda item: item["id"])

    catalog._validate_document(document)


def test_catalog_validator_rejects_unreferenced_templates_and_root_baggage() -> None:
    document = copy.deepcopy(catalog._DOCUMENT)
    template = copy.deepcopy(document["analysis_templates"][0])
    template["id"] = "agentic.unreferenced-probe"
    document["analysis_templates"].append(template)
    document["analysis_templates"].sort(key=lambda item: item["id"])
    with pytest.raises(RuntimeError, match="referenced exhaustively"):
        catalog._validate_document(document)

    document = copy.deepcopy(catalog._DOCUMENT)
    document["legacy_metric_inventory"] = []
    with pytest.raises(RuntimeError, match="fields are invalid"):
        catalog._validate_document(document)


def test_runtime_resolver_rejects_ambiguous_dynamic_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        catalog,
        "_COMPILED_FAMILIES",
        (*catalog._COMPILED_FAMILIES, catalog._COMPILED_FAMILIES[0]),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        catalog.resolve_metric_analysis("capacity.level.16.success_rate")


def _producer_static_metric_ids(repo: Path) -> set[str]:
    metric_pattern = re.compile(
        r"^(?:routing|model_pool|joint|agentic|multimodal|preference|safety|capacity|experiment|r2|routing_recipe)\.[A-Za-z0-9_.-]+$"
    )
    python_root = repo / "src/vllm-sr/cli/evaluation"
    result: set[str] = set()
    for path in (
        *python_root.glob("metric_*.py"),
        python_root / "production_experiment_metric_specs.py",
    ):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and metric_pattern.fullmatch(node.value)
                and not node.value.endswith(".")
            ):
                result.add(node.value)
    go_root = repo / "dashboard/backend/evaluationplane"
    for name in (
        "model_pool_metric_attestation.go",
        "routing_recipe_reducer.go",
        "method_metric_attestation.go",
        "record_metric_attestation.go",
    ):
        source = (go_root / name).read_text(encoding="utf-8")
        result.update(
            value
            for value in re.findall(r'"([^"\\]+)"', source)
            if metric_pattern.fullmatch(value) and not value.endswith(".")
        )
    return result


def test_catalog_exact_ids_match_all_current_metric_producers() -> None:
    repo = Path(__file__).resolve().parents[3]
    assert _producer_static_metric_ids(repo) == set(catalog.STATIC_METRIC_IDS)


def test_evaluation_resources_are_declared_and_present_in_a_built_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setuptools_build = pytest.importorskip("setuptools.build_meta")
    project = Path(__file__).resolve().parents[1]
    staging = tmp_path / "wheel-source"
    staging.mkdir()
    shutil.copy2(project / "pyproject.toml", staging / "pyproject.toml")
    shutil.copy2(project / "README.md", staging / "README.md")
    shutil.copytree(project / "cli", staging / "cli")
    output = tmp_path / "wheel-output"
    output.mkdir()
    monkeypatch.chdir(staging)
    wheel_name = setuptools_build.build_wheel(str(output))
    with zipfile.ZipFile(output / wheel_name) as wheel:
        packaged = wheel.read("cli/evaluation/golden/metric_analysis_catalog.v1.json")
        live_mom_cases = wheel.read("cli/evaluation/resources/live_mom_cases.v1.json")
    assert packaged == catalog.metric_analysis_catalog_bytes()
    assert (
        live_mom_cases
        == (project / "cli/evaluation/resources/live_mom_cases.v1.json").read_bytes()
    )
