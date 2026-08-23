import importlib
import pathlib
import sys

TEST_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(TEST_DIR.parents[1] / "vllm-sr"))

result_to_config = importlib.import_module("result_to_config")
from cli.models import UserConfig  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402

EXPECTED_PHI4_QUALITY_SCORE = 0.775
EXPECTED_QWEN3_QUALITY_SCORE = 0.76
EXPECTED_SIMILARITY_THRESHOLD = 0.85


def test_parse_args_defaults_to_eval_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["result_to_config.py"])
    args = result_to_config.parse_args()
    assert args.output_file == "config/config.eval.yaml"
    assert args.endpoint == "http://127.0.0.1:8000/v1"
    assert args.provider == "openai-compatible"


def test_generate_config_yaml_emits_human_v04_layout():
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
        endpoint="http://127.0.0.1:9000/v1",
        provider="openai-compatible",
    )

    assert set(config) == {
        "version",
        "listeners",
        "models",
        "recipes",
        "entrypoints",
        "global",
    }
    assert config["version"] == "v0.4"
    assert config["listeners"] == []

    generated_models = {model["name"]: model for model in config["models"]}
    assert set(generated_models) == {"phi4", "qwen3-8b"}
    assert generated_models["phi4"]["connections"] == [
        {
            "provider": "openai-compatible",
            "endpoint": "http://127.0.0.1:9000/v1",
            "model": "phi4",
        }
    ]
    assert (
        generated_models["phi4"]["card"]["quality_score"] == EXPECTED_PHI4_QUALITY_SCORE
    )
    assert (
        generated_models["qwen3-8b"]["card"]["quality_score"]
        == EXPECTED_QWEN3_QUALITY_SCORE
    )

    domains = {
        domain["name"]: domain
        for domain in config["recipes"][0]["document"]["signals"]["domains"]
    }
    assert domains["math"]["mmlu_categories"] == ["math"]
    assert domains["law"]["mmlu_categories"] == ["law"]

    decisions = {
        decision["name"]: decision
        for decision in config["recipes"][0]["document"]["decisions"]
    }
    assert set(decisions) == {"default", "law", "math"}
    assignments = config["entrypoints"][0]["assignments"]
    assert assignments["math"] == {"models": [{"model": "qwen3-8b"}]}
    assert assignments["law"] == {"models": [{"model": "phi4"}]}
    assert assignments["default"] == {"models": [{"model": "phi4"}]}
    assert config["entrypoints"][0] == {
        "name": "vllm-sr/eval",
        "aliases": ["eval"],
        "recipe": "mmlu-evaluation",
        "assignments": assignments,
    }

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

    parsed = UserConfig.model_validate(config)
    assert validate_user_config(parsed) == []
