import importlib
import json
import pathlib
import sys

import jsonschema
import yaml

TEST_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

result_to_config = importlib.import_module("result_to_config")

EXPECTED_SIMILARITY_THRESHOLD = 0.85


def test_parse_args_defaults_to_eval_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["result_to_config.py"])
    args = result_to_config.parse_args()
    assert args.output_file == "config/config.eval.yaml"
    assert args.backend_endpoint == "127.0.0.1:8000"
    assert args.backend_protocol == "http"
    assert args.provider_id == "vllm"
    assert args.api_format == "openai"


def test_generate_config_yaml_emits_canonical_v03_layout():
    category_accuracies = {
        "math": {
            "qwen3-8b": 0.82,
            "phi4": 0.74,
            "auto": 0.99,
        },
        "law": {
            "phi4": 0.81,
            "qwen3-8b": 0.70,
        },
    }

    config = result_to_config.generate_config_yaml(
        category_accuracies=category_accuracies,
        similarity_threshold=0.85,
        backend_endpoint="127.0.0.1:9000",
        backend_protocol="http",
        provider_id="vllm",
        api_format="openai",
        provider_name="openai",
    )

    assert set(config) == {"version", "listeners", "providers", "routing", "global"}
    assert config["version"] == "v0.3"
    assert config["listeners"] == []

    defaults = config["providers"]["defaults"]
    assert defaults["model"] == "phi4"
    assert defaults["reasoning_effort"] == "medium"

    provider_models = {model["name"]: model for model in config["providers"]["models"]}
    assert set(provider_models) == {"phi4", "qwen3-8b"}
    assert provider_models["phi4"]["backend_refs"][0]["endpoint"] == "127.0.0.1:9000"
    assert provider_models["phi4"]["backend_refs"][0]["provider"] == "vllm"
    assert provider_models["phi4"]["external_model_ids"] == {"openai": "phi4"}

    routing_models = {model["name"]: model for model in config["routing"]["modelCards"]}
    assert set(routing_models) == {"phi4", "qwen3-8b"}
    assert all("evaluations" not in card for card in routing_models.values())

    domains = {
        domain["name"]: domain for domain in config["routing"]["signals"]["domains"]
    }
    math_scores = domains["math"]["model_scores"]
    assert math_scores[0] == {
        "model": "qwen3-8b",
        "score": 0.82,
        "use_reasoning": True,
    }
    assert domains["law"]["model_scores"][0]["use_reasoning"] is False
    assert config["routing"]["decisions"] == []

    assert (
        config["global"]["stores"]["response_cache"]["similarity_threshold"]
        == EXPECTED_SIMILARITY_THRESHOLD
    )
    assert (
        config["global"]["model_catalog"]["modules"]["classifier"]["domain"][
            "fallback_category"
        ]
        == "other"
    )

    for legacy_key in (
        "default_model",
        "model_config",
        "vllm_endpoints",
        "provider_profiles",
        "categories",
        "decisions",
        "prompt_guard",
        "classifier",
    ):
        assert legacy_key not in config

    root = TEST_DIR.parents[2]
    schema = json.loads(
        (
            root / "src/semantic-router/pkg/configschema/router-config-v0.3.schema.json"
        ).read_text()
    )
    jsonschema.validate(config, schema)
    served = yaml.safe_load((root / "config/config.yaml").read_text())
    actual = config["global"]["model_catalog"]
    expected = served["global"]["model_catalog"]
    guard = actual["modules"]["prompt_guard"]
    for key in (
        "model_id",
        "threshold",
        "variant",
        "positive_labels",
        "jailbreak_mapping_path",
    ):
        assert guard[key] == expected["modules"]["prompt_guard"][key]
    for role, mapping in (
        ("domain", "category_mapping_path"),
        ("pii", "pii_mapping_path"),
    ):
        classifier = actual["modules"]["classifier"][role]
        default = expected["modules"]["classifier"][role]
        assert classifier["model_id"] == default["model_id"]
        assert classifier[mapping] == default[mapping]
    assert actual["embeddings"]["semantic"]["mmbert_model_path"] == (
        expected["embeddings"]["semantic"]["mmbert_model_path"]
    )
