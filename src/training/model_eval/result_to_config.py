# analyze_mmlu_results.py - Analyzes MMLU-Pro results and generates optimized config.yaml

import argparse
import glob
import json
import os
from collections import defaultdict

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MMLU-Pro results and generate optimized config.yaml"
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
        default="config/config.yaml",
        help="Output file for the config.yaml",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help="Similarity threshold for semantic cache",
    )
    return parser.parse_args()


def collect_model_accuracies(results_dir):
    """Collect all model accuracies by category from result files."""
    # Dictionary to store category accuracies for each model
    category_accuracies = defaultdict(lambda: defaultdict(float))

    # Find all analysis.json files
    analysis_files = glob.glob(
        os.path.join(results_dir, "**/analysis.json"), recursive=True
    )

    for file_path in analysis_files:
        # Extract model name from directory path
        dir_name = os.path.basename(os.path.dirname(file_path))
        # Separate model name from approach (cot or direct)
        if "_cot" in dir_name:
            model_name = dir_name.replace("_cot", "")
            approach = "cot"
        else:
            model_name = dir_name.replace("_direct", "")
            approach = "direct"

        # Convert underscores back to slashes in model name
        model_name = (
            model_name.replace("_", "/", 1) if "_" in model_name else model_name
        )
        model_display_name = f"{model_name}:{approach}"

        # Load analysis data
        with open(file_path, "r") as f:
            analysis = json.load(f)

        # Store category accuracies
        for category, accuracy in analysis.get("category_accuracy", {}).items():
            category_accuracies[category][model_display_name] = accuracy

    return category_accuracies


def generate_config_yaml(category_accuracies, similarity_threshold):
    """Generate config.yaml with models ranked by performance for each category."""
    # Prepare the base configuration
    config = {
        "bert_model": {
            "model_id": "sentence-transformers/all-MiniLM-L12-v2",
            "threshold": 0.6,
            "use_cpu": True,
        },
        "semantic_cache": {
            "enabled": True,
            "similarity_threshold": similarity_threshold,
            "max_entries": 1000,
            "ttl_seconds": 3600,
        },
        "tools": {
            "enabled": True,
            "top_k": 3,
            "similarity_threshold": 0.2,
            "tools_db_path": "config/tools_db.json",
            "fallback_to_empty": True,
        },
        "prompt_guard": {
            "enabled": True,
            "use_modernbert": True,
            "model_id": "models/mom-jailbreak-classifier",
            "threshold": 0.7,
            "use_cpu": True,
            "jailbreak_mapping_path": "models/mom-jailbreak-classifier/jailbreak_type_mapping.json",
        },
        "classifier": {
            "category_model": {
                "model_id": "models/mom-domain-classifier",
                "use_modernbert": True,
                "threshold": 0.6,
                "use_cpu": True,
                "category_mapping_path": "models/mom-domain-classifier/category_mapping.json",
            },
            "pii_model": {
                "model_id": "models/pii_classifier_modernbert-base_presidio_token_model",
                "use_modernbert": True,
                "threshold": 0.7,
                "use_cpu": True,
                "pii_mapping_path": "models/mom-pii-classifier/pii_type_mapping.json",
            },
        },
        "categories": [],
        "default_reasoning_effort": "medium",  # Default reasoning effort level (low, medium, high)
    }

    # Get the best model overall to use as default (excluding 'auto')
    all_models_avg = defaultdict(list)
    for category, models in category_accuracies.items():
        for model_name, accuracy in models.items():
            base_model = model_name.split(":")[0]
            if base_model != "auto":
                all_models_avg[model_name].append(accuracy)

    # Calculate average accuracy across all categories
    model_avg_accuracies = {
        model: sum(accuracies) / len(accuracies)
        for model, accuracies in all_models_avg.items()
    }

    # Set default model to the one with highest average accuracy
    default_model = max(model_avg_accuracies, key=model_avg_accuracies.get)
    config["default_model"] = default_model.split(":")[0]  # Remove the approach suffix

    # Define category-specific reasoning settings
    category_reasoning = {
        "math": {
            "use_reasoning": True,
            "reasoning_description": "Mathematical problems require step-by-step reasoning",
            "reasoning_effort": "high",
        },
        "physics": {
            "use_reasoning": True,
            "reasoning_description": "Physics concepts need logical analysis",
            "reasoning_effort": "high",
        },
        "chemistry": {
            "use_reasoning": True,
            "reasoning_description": "Chemical reactions and formulas require systematic thinking",
            "reasoning_effort": "high",
        },
        "computer science": {
            "use_reasoning": True,
            "reasoning_description": "Programming and algorithms need logical reasoning",
            "reasoning_effort": "high",
        },
        "engineering": {
            "use_reasoning": True,
            "reasoning_description": "Engineering problems require systematic problem-solving",
            "reasoning_effort": "high",
        },
        "biology": {
            "use_reasoning": True,
            "reasoning_description": "Biological processes benefit from structured analysis",
            "reasoning_effort": "medium",
        },
        "business": {
            "use_reasoning": False,
            "reasoning_description": "Business content is typically conversational",
            "reasoning_effort": "low",
        },
        "law": {
            "use_reasoning": False,
            "reasoning_description": "Legal content is typically explanatory",
            "reasoning_effort": "medium",
        },
        "psychology": {
            "use_reasoning": False,
            "reasoning_description": "Psychology content is usually explanatory",
            "reasoning_effort": "medium",
        },
        "history": {
            "use_reasoning": False,
            "reasoning_description": "Historical content is narrative-based",
            "reasoning_effort": "low",
        },
        "economics": {
            "use_reasoning": False,
            "reasoning_description": "Economic discussions are usually explanatory",
            "reasoning_effort": "medium",
        },
        "philosophy": {
            "use_reasoning": False,
            "reasoning_description": "Philosophical discussions are conversational",
            "reasoning_effort": "medium",
        },
        "health": {
            "use_reasoning": False,
            "reasoning_description": "Health information is typically informational",
            "reasoning_effort": "medium",
        },
        "other": {
            "use_reasoning": False,
            "reasoning_description": "General content doesn't require reasoning",
            "reasoning_effort": "low",
        },
    }

    # Create category entries with ranked model-score pairs (excluding 'auto')
    for category, models in category_accuracies.items():
        # Sort models by accuracy (descending), exclude 'auto'
        ranked_models = [
            (model.split(":")[0], acc)
            for model, acc in sorted(models.items(), key=lambda x: x[1], reverse=True)
            if model.split(":")[0] != "auto"
        ]
        # Get reasoning settings for the category
        reasoning_settings = category_reasoning.get(
            category.lower(),
            {
                "use_reasoning": False,
                "reasoning_description": "General content doesn't require reasoning",
                "reasoning_effort": "low",
            },
        )
        # Build the model_scores list with reasoning settings applied to each model
        model_scores = []
        for model, acc in ranked_models:
            model_score = {
                "model": model,
                "score": float(acc),
                "use_reasoning": reasoning_settings["use_reasoning"],
                "reasoning_description": reasoning_settings["reasoning_description"],
                "reasoning_effort": reasoning_settings["reasoning_effort"],
            }
            model_scores.append(model_score)

        # Add category to config
        config["categories"].append(
            {
                "name": category,
                "model_scores": model_scores,
            }
        )

    return config


def save_config(config, output_file):
    """Save the config dictionary as a YAML file."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config saved to {output_file}")


def main():
    args = parse_args()

    print(f"Analyzing MMLU-Pro results in {args.results_dir}...")
    category_accuracies = collect_model_accuracies(args.results_dir)

    print(f"Generating config.yaml...")
    config = generate_config_yaml(category_accuracies, args.similarity_threshold)

    print(f"Saving config to {args.output_file}...")
    save_config(config, args.output_file)

    print("Done!")


if __name__ == "__main__":
    main()
