"""The removed model-resource and model-runtime forms, and the layouts before
config v0.3, as the CLI reads them: refused with the Router's own lines, and
gone after `vllm-sr config migrate`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_legacy_layout_forms import (  # noqa: E402
    removed_legacy_router_forms,
    removed_previous_release_forms,
)
from cli.config_model_resource_forms import removed_model_resource_forms  # noqa: E402
from cli.config_model_runtime_forms import removed_model_runtime_forms  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config_data  # noqa: E402

GOLDEN = REPO_ROOT / "src/semantic-router/pkg/config/testdata/migrate"
CASES = (
    "removed-field-deployment-fields",
    "invalid-value-classifier-provider",
    "removed-field-classifier-binding",
    "removed-field-binding-object",
    "removed-field-rule-binding",
    "removed-section-external-models",
    "removed-section-external-models-empty",
    "removed-field-backend-connection",
    "invalid-value-model-provider",
    "removed-field-runtime-profile",
    "removed-field-nli-explainer",
    "removed-field-router-names",
    "removed-field-router-names-beside-entrypoint",
    "removed-field-router-names-single",
    "removed-field-deployment-operating-point",
    "removed-field-looper-model-names",
    "removed-field-looper-model-names-unserved",
    "removed-section-pre-v03-layout",
    "invalid-value-pre-v03-version",
    "removed-field-pre-v03-provider-catalog",
    "removed-section-pre-v03-mom-registry",
    "removed-section-pre-v03-prompt-compression",
    "invalid-value-classifier-provider-references",
    "invalid-value-classifier-provider-decision-model",
)
MIGRATED = tuple(case for case in CASES if (GOLDEN / case / "after.yaml").exists())


def _golden(case: str, name: str) -> dict:
    return yaml.safe_load((GOLDEN / case / name).read_text(encoding="utf-8"))


def _refusal(document: dict) -> list[str]:
    with pytest.raises(ConfigParseError) as refused:
        parse_user_config_data(document, log_summary=False)
    return str(refused.value).splitlines()


@pytest.mark.parametrize("case", CASES)
def test_parser_refuses_a_removed_model_form_with_the_router_line(case: str) -> None:
    hint = (GOLDEN / case / "hint.txt").read_text(encoding="utf-8").strip()

    assert hint in _refusal(_golden(case, "before.yaml"))


@pytest.mark.parametrize("case", MIGRATED)
def test_migrated_model_forms_have_no_removed_forms(case: str) -> None:
    after = _golden(case, "after.yaml")

    assert removed_previous_release_forms(after) == []
    assert removed_legacy_router_forms(after) == []
    assert removed_model_runtime_forms(after) == []
    assert removed_model_resource_forms(after) == []


def _canonical(**extra) -> dict:
    return {
        "version": "v0.5",
        "providers": {
            "defaults": {"model": "m"},
            "models": [{"name": "m", "backend_refs": [{"endpoint": "m:8000"}]}],
        },
        "recipes": [{"name": "default"}],
        **extra,
    }


PREVIOUS_RELEASE = "run `vllm-sr config migrate` (its place in the v0.5 layout)"


def test_the_layouts_before_config_v03_are_refused_with_the_router_lines():
    document = _canonical(keyword_rules=[{"name": "k", "keywords": ["hello"]}])
    document["global"] = {"modules": {"prompt_guard": {"model_ref": "guard"}}}
    document["providers"]["models"][0]["access_key"] = "${KEY}"

    assert _refusal(document)[:3] == [
        "keyword_rules: removed_section: top-level keyword_rules belongs to the "
        f"layout before config v0.3; {PREVIOUS_RELEASE}",
        "global.modules: removed_field: global.modules belongs to the global "
        f"layout before config v0.3; {PREVIOUS_RELEASE}",
        "providers.models[m].access_key: removed_field: "
        "providers.models[m].access_key belongs to the provider catalog before "
        f"config v0.3; {PREVIOUS_RELEASE}",
    ]


def test_the_looper_endpoint_and_model_names_are_refused_with_the_router_lines():
    looper = {"endpoint": "http://x/v1", "fusion": {"model_names": ["fused"]}}
    document = _canonical(**{"global": {"integrations": {"looper": looper}}})

    assert _refusal(document) == [
        "global.integrations.looper.endpoint: removed_field: the Looper endpoint "
        "is removed; run `vllm-sr config migrate` (nothing; the Router makes a "
        "Looper's model calls in process, through each model's "
        "providers.models[].backend_refs)",
        "global.integrations.looper.fusion.model_names: removed_field: the model "
        "names of a Looper algorithm are removed; run `vllm-sr config migrate` (an "
        "entrypoint whose model_names name a recipe with a decision that runs the "
        "fusion algorithm)",
    ]
