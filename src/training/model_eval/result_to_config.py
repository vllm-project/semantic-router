"""Analyze MMLU-Pro results and generate a canonical v0.3 config scaffold."""

import argparse
import glob
import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import yaml


def artifact_contract_path():
    """Return the shared training artifact contract path."""
    return (
        Path(__file__).resolve().parents[2]
        / "semantic-router"
        / "pkg"
        / "trainingartifacts"
        / "contract.json"
    )


def load_artifact_contract():
    """Load the shared training artifact contract."""
    with artifact_contract_path().open(encoding="utf-8") as handle:
        return json.load(handle)


SHARED_ARTIFACT_CONTRACT = load_artifact_contract()
RUNTIME_DEFAULTS = SHARED_ARTIFACT_CONTRACT["runtime_defaults"]
DEFAULT_OUTPUT_FILE = SHARED_ARTIFACT_CONTRACT["model_eval"]["outputs"][
    "default_config_output_file"
]
DEFAULT_REASONING_EFFORT = RUNTIME_DEFAULTS["providers"]["default_reasoning_effort"]
DEFAULT_EMBEDDINGS = RUNTIME_DEFAULTS["embeddings"]
DEFAULT_SEMANTIC_CACHE = RUNTIME_DEFAULTS["semantic_cache"]
DEFAULT_TOOLS = RUNTIME_DEFAULTS["tools"]
DEFAULT_PROMPT_GUARD = RUNTIME_DEFAULTS["prompt_guard"]
DEFAULT_DOMAIN_CLASSIFIER = RUNTIME_DEFAULTS["domain_classifier"]
DEFAULT_PII_CLASSIFIER = RUNTIME_DEFAULTS["pii_classifier"]
CATEGORY_REASONING = RUNTIME_DEFAULTS["category_reasoning"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MMLU-Pro results and generate a canonical v0.3 config scaffold"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing MMLU-Pro results",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output file for the generated canonical config scaffold",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help="Similarity threshold for the generated semantic cache override",
    )
    parser.add_argument(
        "--backend-endpoint",
        type=str,
        default="127.0.0.1:8000",
        help="Endpoint to bind generated providers.models[].backend_refs[] entries to",
    )
    parser.add_argument(
        "--backend-protocol",
        type=str,
        default="http",
        help="Protocol for generated backend_refs entries",
    )
    parser.add_argument(
        "--backend-type",
        type=str,
        default="chat",
        help="Backend type for generated backend_refs entries",
    )
    parser.add_argument(
        "--api-format",
        type=str,
        default="openai",
        help="API format for generated providers.models entries",
    )
    parser.add_argument(
        "--provider-name",
        type=str,
        default="openai",
        help="Provider name used in generated external_model_ids",
    )
    return parser.parse_args()


def collect_model_accuracies(results_dir):
    """Collect all model accuracies by category from result files."""
    category_accuracies = defaultdict(lambda: defaultdict(float))
    analysis_files = glob.glob(
        os.path.join(results_dir, "**/analysis.json"), recursive=True
    )

    for file_path in analysis_files:
        dir_name = os.path.basename(os.path.dirname(file_path))
        if "_cot" in dir_name:
            model_name = dir_name.replace("_cot", "")
        else:
            model_name = dir_name.replace("_direct", "")

        model_name = (
            model_name.replace("_", "/", 1) if "_" in model_name else model_name
        )

        with open(file_path, encoding="utf-8") as handle:
            analysis = json.load(handle)

        for category, accuracy in analysis.get("category_accuracy", {}).items():
            category_accuracies[category][model_name] = max(
                category_accuracies[category][model_name],
                float(accuracy),
            )

    return category_accuracies


def calculate_average_accuracies(category_accuracies):
    """Compute average per-model accuracy across categories after variant collapse."""
    averages = defaultdict(list)
    for models in category_accuracies.values():
        for model_name, accuracy in models.items():
            if model_name == "auto":
                continue
            averages[model_name].append(float(accuracy))

    return {
        model_name: sum(scores) / len(scores)
        for model_name, scores in averages.items()
        if scores
    }


def build_provider_models(
    ranked_models,
    backend_endpoint,
    backend_protocol,
    backend_type,
    api_format,
    provider_name,
):
    provider_models = []
    for model_name, _average_accuracy in ranked_models:
        provider_models.append(
            {
                "name": model_name,
                "provider_model_id": model_name,
                "api_format": api_format,
                "external_model_ids": {provider_name: model_name},
                "backend_refs": [
                    {
                        "name": f"{model_name}-backend",
                        "endpoint": backend_endpoint,
                        "protocol": backend_protocol,
                        "type": backend_type,
                        "weight": 1,
                    }
                ],
            }
        )
    return provider_models


def build_routing_model_cards(ranked_models):
    model_cards = []
    for model_name, average_accuracy in ranked_models:
        model_cards.append(
            {
                "name": model_name,
                "description": (
                    "Generated from MMLU-Pro evaluation results for category-aware routing."
                ),
                "quality_score": round(float(average_accuracy), 6),
                "capabilities": ["chat"],
                "tags": ["generated", "mmlu-pro"],
                "modality": "ar",
            }
        )
    return model_cards


def build_domain_signals(category_accuracies):
    domains = []
    for category_name, models in sorted(category_accuracies.items()):
        ranked_models = sorted(
            (
                (model_name, float(accuracy))
                for model_name, accuracy in models.items()
                if model_name != "auto"
            ),
            key=lambda item: (-item[1], item[0]),
        )
        domains.append(
            {
                "name": category_name,
                "description": (
                    f"MMLU-Pro category generated from evaluation results: {category_name}."
                ),
                "mmlu_categories": [category_name],
                "model_scores": [
                    {
                        "model": model_name,
                        "score": round(accuracy, 6),
                        "use_reasoning": CATEGORY_REASONING.get(
                            category_name.lower(), False
                        ),
                    }
                    for model_name, accuracy in ranked_models
                ],
            }
        )
    return domains


def generate_config_yaml(
    category_accuracies,
    similarity_threshold,
    backend_endpoint,
    backend_protocol,
    backend_type,
    api_format,
    provider_name,
):
    """Generate a canonical v0.3 config scaffold from MMLU-Pro results."""
    average_accuracies = calculate_average_accuracies(category_accuracies)
    if not average_accuracies:
        raise ValueError("No non-auto model results were found in the input directory")

    ranked_models = sorted(
        average_accuracies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    default_model = ranked_models[0][0]

    return {
        "version": "v0.3",
        "listeners": [],
        "providers": {
            "defaults": {
                "default_model": default_model,
                "default_reasoning_effort": DEFAULT_REASONING_EFFORT,
            },
            "models": build_provider_models(
                ranked_models,
                backend_endpoint,
                backend_protocol,
                backend_type,
                api_format,
                provider_name,
            ),
        },
        "routing": {
            "modelCards": build_routing_model_cards(ranked_models),
            "signals": {
                "domains": build_domain_signals(category_accuracies),
            },
            "decisions": [],
        },
        "global": {
            "stores": {
                "semantic_cache": {
                    **deepcopy(DEFAULT_SEMANTIC_CACHE),
                    "similarity_threshold": similarity_threshold,
                }
            },
            "integrations": {
                "tools": deepcopy(DEFAULT_TOOLS),
            },
            "model_catalog": {
                "embeddings": deepcopy(DEFAULT_EMBEDDINGS),
                "modules": {
                    "prompt_guard": deepcopy(DEFAULT_PROMPT_GUARD),
                    "classifier": {
                        "domain": deepcopy(DEFAULT_DOMAIN_CLASSIFIER),
                        "pii": deepcopy(DEFAULT_PII_CLASSIFIER),
                    },
                },
            },
        },
    }


def save_config(config, output_file):
    """Save the config dictionary as a YAML file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=False)

    print(f"Config saved to {output_file}")


def main():
    args = parse_args()

    print(f"Analyzing MMLU-Pro results in {args.results_dir}...")
    category_accuracies = collect_model_accuracies(args.results_dir)

    print("Generating canonical v0.3 config scaffold...")
    config = generate_config_yaml(
        category_accuracies,
        args.similarity_threshold,
        args.backend_endpoint,
        args.backend_protocol,
        args.backend_type,
        args.api_format,
        args.provider_name,
    )

    print(f"Saving config to {args.output_file}...")
    save_config(config, args.output_file)

    print("Done!")


if __name__ == "__main__":
    main()
