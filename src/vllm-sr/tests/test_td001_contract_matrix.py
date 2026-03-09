# ruff: noqa: E402

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.authoring_projection import build_first_slice_authoring_config
from cli.authoring_runtime_compile import (
    build_first_slice_runtime_overlay,
    compile_first_slice_runtime,
)
from cli.dashboard_bridge import (
    load_dashboard_config,
    merge_dashboard_config_patch,
    render_dashboard_yaml,
)
from cli.defaults import load_embedded_defaults
from cli.merger import merge_configs
from cli.parser import parse_user_config
from cli.validator import validate_merged_config, validate_user_config

AUTHORING_FIXTURE = (
    REPO_ROOT / "config" / "testing" / "td001-first-slice-authoring.yaml"
)
RUNTIME_FIXTURE = REPO_ROOT / "config" / "testing" / "td001-first-slice-runtime.yaml"
FIRST_SLICE_LEGACY_RUNTIME_KEYS = {
    "default_model",
    "default_reasoning_effort",
    "keyword_rules",
    "model_config",
    "reasoning_families",
    "vllm_endpoints",
}


def _load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_config_dict(config_data: dict):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config_data, f, sort_keys=False)
        config_path = f.name

    try:
        return parse_user_config(config_path)
    finally:
        Path(config_path).unlink(missing_ok=True)


def _extract_model_config_slice(model_config):
    extracted = {}
    for model_name, config in model_config.items():
        entry = {}
        for key in [
            "preferred_endpoints",
            "reasoning_family",
            "access_key",
            "param_size",
            "api_format",
            "description",
            "capabilities",
            "quality_score",
            "pricing",
        ]:
            value = config.get(key)
            if value is None:
                continue
            if value in ("", [], {}):
                continue
            entry[key] = copy.deepcopy(value)
        extracted[model_name] = entry
    return extracted


def _extract_decision_slice(decisions):
    extracted = []
    for decision in decisions:
        entry = {
            "name": decision["name"],
            "rules": copy.deepcopy(decision["rules"]),
            "modelRefs": [],
        }
        if decision.get("description"):
            entry["description"] = decision["description"]
        if decision.get("priority") is not None:
            entry["priority"] = decision["priority"]

        for model_ref in decision.get("modelRefs", []):
            ref_entry = {
                "model": model_ref["model"],
                "use_reasoning": model_ref["use_reasoning"],
            }
            if model_ref.get("reasoning_effort"):
                ref_entry["reasoning_effort"] = model_ref["reasoning_effort"]
            if model_ref.get("lora_name"):
                ref_entry["lora_name"] = model_ref["lora_name"]
            entry["modelRefs"].append(ref_entry)

        extracted.append(entry)
    return extracted


def _extract_runtime_slice(merged):
    return {
        "listeners": copy.deepcopy(merged["listeners"]),
        "keyword_rules": copy.deepcopy(merged["keyword_rules"]),
        "vllm_endpoints": copy.deepcopy(merged["vllm_endpoints"]),
        "model_config": _extract_model_config_slice(merged["model_config"]),
        "default_model": merged["default_model"],
        "reasoning_families": copy.deepcopy(merged["reasoning_families"]),
        "default_reasoning_effort": merged["default_reasoning_effort"],
        "decisions": _extract_decision_slice(merged["decisions"]),
    }


class TestTD001ContractMatrix(unittest.TestCase):
    maxDiff = None

    def test_cli_projects_shared_first_slice_authoring_fixture(self):
        user_config = parse_user_config(str(AUTHORING_FIXTURE))

        self.assertEqual([], validate_user_config(user_config))
        self.assertEqual(
            _load_yaml(AUTHORING_FIXTURE),
            build_first_slice_authoring_config(user_config),
        )

    def test_cli_compiles_shared_first_slice_authoring_fixture_to_runtime_fixture(self):
        self.assertEqual(
            _load_yaml(RUNTIME_FIXTURE),
            compile_first_slice_runtime(_load_yaml(AUTHORING_FIXTURE)),
        )

    def test_cli_merge_matches_shared_first_slice_runtime_fixture(self):
        user_config = parse_user_config(str(AUTHORING_FIXTURE))

        user_errors = validate_user_config(user_config)
        self.assertEqual([], user_errors)

        merged = merge_configs(user_config, load_embedded_defaults())

        merged_errors = validate_merged_config(merged)
        self.assertEqual([], merged_errors)

        self.assertEqual(_load_yaml(RUNTIME_FIXTURE), _extract_runtime_slice(merged))

    def test_cli_dual_read_normalizes_shared_first_slice_runtime_fixture(self):
        user_config = parse_user_config(str(RUNTIME_FIXTURE))

        self.assertEqual("v0.1", user_config.version)
        self.assertEqual([], validate_user_config(user_config))

        extra = getattr(user_config, "model_extra", {}) or {}
        self.assertFalse(FIRST_SLICE_LEGACY_RUNTIME_KEYS & set(extra))

        merged = merge_configs(user_config, load_embedded_defaults())

        self.assertEqual([], validate_merged_config(merged))
        self.assertEqual(_load_yaml(RUNTIME_FIXTURE), _extract_runtime_slice(merged))

    def test_dashboard_bridge_loads_runtime_fixture_into_canonical_shape(self):
        loaded_dashboard_config = load_dashboard_config(str(RUNTIME_FIXTURE))
        loaded_user_config = _parse_config_dict(loaded_dashboard_config)

        self.assertFalse(FIRST_SLICE_LEGACY_RUNTIME_KEYS & set(loaded_dashboard_config))
        self.assertEqual(
            _load_yaml(AUTHORING_FIXTURE),
            build_first_slice_authoring_config(loaded_user_config),
        )

    def test_dashboard_bridge_renders_canonical_yaml_from_authoring_fixture(self):
        rendered = render_dashboard_yaml(_load_yaml(AUTHORING_FIXTURE))
        rendered_dashboard_config = yaml.safe_load(rendered)
        rendered_user_config = _parse_config_dict(rendered_dashboard_config)

        self.assertFalse(
            FIRST_SLICE_LEGACY_RUNTIME_KEYS & set(rendered_dashboard_config)
        )
        self.assertEqual(
            _load_yaml(AUTHORING_FIXTURE),
            build_first_slice_authoring_config(rendered_user_config),
        )

    def test_dashboard_bridge_merges_partial_canonical_patch_without_legacy_keys(self):
        merged_dashboard_config = merge_dashboard_config_patch(
            load_dashboard_config(str(RUNTIME_FIXTURE)),
            {
                "providers": {
                    "default_reasoning_effort": "high",
                }
            },
        )
        merged_user_config = _parse_config_dict(merged_dashboard_config)

        self.assertFalse(FIRST_SLICE_LEGACY_RUNTIME_KEYS & set(merged_dashboard_config))
        self.assertEqual(
            "high",
            merged_dashboard_config["providers"]["default_reasoning_effort"],
        )
        self.assertEqual([], validate_user_config(merged_user_config))

    def test_dashboard_bridge_merges_typed_compat_patch(self):
        merged_dashboard_config = merge_dashboard_config_patch(
            load_dashboard_config(str(AUTHORING_FIXTURE)),
            {
                "tools": {
                    "enabled": True,
                    "top_k": 7,
                }
            },
        )

        self.assertEqual(7, merged_dashboard_config["tools"]["top_k"])
        self.assertFalse(FIRST_SLICE_LEGACY_RUNTIME_KEYS & set(merged_dashboard_config))
        self.assertEqual(
            [], validate_user_config(_parse_config_dict(merged_dashboard_config))
        )

    def test_cli_first_slice_runtime_overlay_preserves_extensions(self):
        user_config = _parse_config_dict(
            {
                "version": "v0.1",
                "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
                "signals": {
                    "keywords": [
                        {
                            "name": "billing_keywords",
                            "operator": "contains",
                            "keywords": ["invoice"],
                        }
                    ]
                },
                "providers": {
                    "models": [
                        {
                            "name": "router-model",
                            "endpoints": [
                                {
                                    "name": "primary",
                                    "weight": 100,
                                    "endpoint": "router.internal:8000",
                                }
                            ],
                        }
                    ],
                    "default_model": "router-model",
                    "external_models": [
                        {
                            "role": "guardrail",
                            "provider": "vllm",
                            "endpoint": "guardrail.internal",
                            "model_name": "guardrail-model",
                        }
                    ],
                },
                "decisions": [
                    {
                        "name": "billing-route",
                        "description": "Route billing questions",
                        "priority": 90,
                        "rules": {
                            "operator": "AND",
                            "conditions": [
                                {"type": "keyword", "name": "billing_keywords"}
                            ],
                        },
                        "modelRefs": [{"model": "router-model"}],
                        "algorithm": {"type": "static"},
                        "plugins": [
                            {
                                "type": "system_prompt",
                                "configuration": {
                                    "enabled": True,
                                    "system_prompt": "Route carefully",
                                    "mode": "replace",
                                },
                            }
                        ],
                    }
                ],
            }
        )

        runtime_overlay = build_first_slice_runtime_overlay(user_config)

        self.assertEqual("static", runtime_overlay["decisions"][0]["algorithm"]["type"])
        self.assertEqual(
            "system_prompt", runtime_overlay["decisions"][0]["plugins"][0]["type"]
        )
        self.assertEqual(
            "guardrail", runtime_overlay["external_models"][0]["model_role"]
        )


if __name__ == "__main__":
    unittest.main()
