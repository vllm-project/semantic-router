# ruff: noqa: E402

import copy
import sys
import unittest
from pathlib import Path

import yaml

CLI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.defaults import load_embedded_defaults
from cli.merger import merge_configs
from cli.parser import parse_user_config
from cli.validator import validate_merged_config, validate_user_config

AUTHORING_FIXTURE = (
    REPO_ROOT / "config" / "testing" / "td001-first-slice-authoring.yaml"
)
RUNTIME_FIXTURE = REPO_ROOT / "config" / "testing" / "td001-first-slice-runtime.yaml"


def _load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


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

    def test_cli_merge_matches_shared_first_slice_runtime_fixture(self):
        user_config = parse_user_config(str(AUTHORING_FIXTURE))

        user_errors = validate_user_config(user_config)
        self.assertEqual([], user_errors)

        merged = merge_configs(user_config, load_embedded_defaults())

        merged_errors = validate_merged_config(merged)
        self.assertEqual([], merged_errors)

        self.assertEqual(_load_yaml(RUNTIME_FIXTURE), _extract_runtime_slice(merged))


if __name__ == "__main__":
    unittest.main()
