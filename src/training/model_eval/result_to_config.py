"""Analyze MMLU-Pro results and generate a human-readable v0.4 scaffold."""

import argparse
import glob
import json
import os
from collections import defaultdict

import yaml

DEFAULT_OUTPUT_FILE = "config/config.eval.yaml"

DEFAULT_EMBEDDINGS = {
    "semantic": {
        "mmbert_model_path": "models/mom-embedding-ultra",
        "use_cpu": True,
        "embedding_config": {
            "model_type": "mmbert",
            "preload_embeddings": True,
            "target_dimension": 768,
            "target_layer": 22,
            "min_score_threshold": 0.5,
        },
    }
}

DEFAULT_RESPONSE_CACHE = {
    "enabled": True,
    "embedding_model": "mmbert",
    "max_entries": 1000,
    "ttl_seconds": 3600,
}

DEFAULT_TOOLS = {
    "enabled": True,
    "top_k": 3,
    "similarity_threshold": 0.2,
    "tools_db_path": "config/runtime/tools/tools_db.json",
    "fallback_to_empty": True,
}

DEFAULT_PROMPT_GUARD = {
    "enabled": True,
    "model_id": "models/mmbert32k-jailbreak-detector-merged",
    "threshold": 0.7,
    "use_cpu": True,
    "use_mmbert_32k": True,
    "jailbreak_mapping_path": (
        "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json"
    ),
}

DEFAULT_DOMAIN_CLASSIFIER = {
    "model_id": "models/mmbert32k-intent-classifier-merged",
    "threshold": 0.5,
    "use_cpu": True,
    "use_mmbert_32k": True,
    "category_mapping_path": "models/mmbert32k-intent-classifier-merged/category_mapping.json",
    "fallback_category": "other",
}

DEFAULT_PII_CLASSIFIER = {
    "model_id": "models/mmbert32k-pii-detector-merged",
    "threshold": 0.9,
    "use_cpu": True,
    "use_mmbert_32k": True,
    "pii_mapping_path": "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MMLU-Pro results and generate a human-readable v0.4 config scaffold"
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
        "--endpoint",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible endpoint shared by the generated Models",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai-compatible",
        help="Provider Integration used by the generated Model connections",
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


def build_models(ranked_models, endpoint, provider):
    """Build concise logical Models from ranked evaluation results."""
    generated_models = []
    for model_name, average_accuracy in ranked_models:
        generated_models.append(
            {
                "name": model_name,
                "card": {
                    "description": (
                        "Generated from MMLU-Pro evaluation results for "
                        "category-aware routing."
                    ),
                    "quality_score": round(float(average_accuracy), 6),
                    "capabilities": ["chat"],
                    "tags": ["generated", "mmlu-pro"],
                    "modality": "text",
                },
                "connections": [
                    {
                        "provider": provider,
                        "endpoint": endpoint,
                        "model": model_name,
                    }
                ],
            }
        )
    return generated_models


def build_domain_signals(category_accuracies):
    domains = []
    for category_name in sorted(category_accuracies):
        domains.append(
            {
                "name": category_name,
                "description": (
                    f"MMLU-Pro category generated from evaluation results: {category_name}."
                ),
                "mmlu_categories": [category_name],
            }
        )
    return domains


def build_decisions_and_assignments(category_accuracies, default_model):
    """Turn each observed category into one readable route and Model assignment."""
    decisions = []
    assignments = {}
    for offset, (category_name, models) in enumerate(
        sorted(category_accuracies.items())
    ):
        candidates = [
            (model_name, float(accuracy))
            for model_name, accuracy in models.items()
            if model_name != "auto"
        ]
        if not candidates:
            continue
        selected_model, _ = min(candidates, key=lambda item: (-item[1], item[0]))
        decisions.append(
            {
                "name": category_name,
                "description": (
                    f"Route {category_name} requests to their best evaluated Model."
                ),
                "priority": 100 - offset,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "domain", "name": category_name}],
                },
            }
        )
        assignments[category_name] = {"models": [{"model": selected_model}]}

    decisions.append(
        {
            "name": "default",
            "description": "Handle requests outside the evaluated categories.",
            "priority": 0,
            "rules": {"operator": "AND", "conditions": []},
        }
    )
    assignments["default"] = {"models": [{"model": default_model}]}
    return decisions, assignments


def generate_config_yaml(
    category_accuracies,
    similarity_threshold,
    endpoint,
    provider,
):
    """Generate a human-readable v0.4 config scaffold from MMLU-Pro results."""
    average_accuracies = calculate_average_accuracies(category_accuracies)
    if not average_accuracies:
        raise ValueError("No non-auto model results were found in the input directory")

    ranked_models = sorted(
        average_accuracies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    default_model = ranked_models[0][0]
    decisions, assignments = build_decisions_and_assignments(
        category_accuracies,
        default_model,
    )

    return {
        "version": "v0.4",
        "listeners": [],
        "models": build_models(ranked_models, endpoint, provider),
        "recipes": [
            {
                "name": "mmlu-evaluation",
                "description": "Category routing derived from MMLU-Pro results.",
                "document": {
                    "signals": {
                        "domains": build_domain_signals(category_accuracies),
                    },
                    "decisions": decisions,
                },
            },
        ],
        "entrypoints": [
            {
                "name": "vllm-sr/eval",
                "aliases": ["eval"],
                "recipe": "mmlu-evaluation",
                "assignments": assignments,
            }
        ],
        "global": {
            "stores": {
                "response_cache": {
                    **DEFAULT_RESPONSE_CACHE,
                    "similarity_threshold": similarity_threshold,
                }
            },
            "integrations": {
                "tools": DEFAULT_TOOLS,
            },
            "model_catalog": {
                "embeddings": DEFAULT_EMBEDDINGS,
                "modules": {
                    "prompt_guard": DEFAULT_PROMPT_GUARD,
                    "classifier": {
                        "domain": DEFAULT_DOMAIN_CLASSIFIER,
                        "pii": DEFAULT_PII_CLASSIFIER,
                    },
                },
            },
            "services": {
                "backend_dispatch": {
                    "bind_address": "127.0.0.1",
                    "port": 8180,
                    "audience": "vllm-sr.backend-dispatch",
                    "capability_ttl": "30s",
                    "max_request_body_bytes": 67108864,
                },
                "backend_egress": {
                    "policy_file": "/app/config/backend-egress-policy.yaml"
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

    print("Generating human-readable v0.4 config scaffold...")
    config = generate_config_yaml(
        category_accuracies,
        args.similarity_threshold,
        args.endpoint,
        args.provider,
    )

    print(f"Saving config to {args.output_file}...")
    save_config(config, args.output_file)

    print("Done!")


if __name__ == "__main__":
    main()
