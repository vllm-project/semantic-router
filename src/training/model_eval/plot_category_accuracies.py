#!/usr/bin/env python3
# Copyright 2025 The vLLM Semantic Router Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# plot_category_accuracies.py - Visualize MMLU-Pro results by category

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MMLU-Pro accuracies by category")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing MMLU-Pro results",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="model_eval/category_accuracies.png",
        help="Output file for the plot",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["bar", "heatmap"],
        default="bar",
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use sample data if no results are found",
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


def generate_sample_data():
    """Generate sample data for demonstration when no results are available."""
    sample_data = defaultdict(lambda: defaultdict(float))

    # Define sample categories and models
    categories = ["Math", "Science", "History", "Literature", "Computer Science"]
    models = [
        "gpt-4:cot",
        "gpt-4:direct",
        "claude-3:cot",
        "claude-3:direct",
        "llama-3:cot",
    ]

    # Generate random accuracies
    np.random.seed(42)  # For reproducibility
    for category in categories:
        for model in models:
            # Generate accuracy between 0.5 and 1.0
            accuracy = 0.5 + np.random.random() * 0.5
            sample_data[category][model] = accuracy

    return sample_data


def add_overall_accuracy(category_accuracies):
    """Add an 'Overall' category showing the average accuracy across all categories."""
    # Create a new dict to avoid modifying the original during iteration
    result = defaultdict(lambda: defaultdict(float))

    # Copy existing data
    for category, models in category_accuracies.items():
        for model, accuracy in models.items():
            result[category][model] = accuracy

    # Calculate average accuracy for each model across all categories
    model_avg_accuracies = defaultdict(list)
    for category, models in category_accuracies.items():
        for model, accuracy in models.items():
            model_avg_accuracies[model].append(accuracy)

    # Add the 'Overall' category with average accuracies
    for model, accuracies in model_avg_accuracies.items():
        if accuracies:  # Ensure there are values to average
            result["Overall"][model] = sum(accuracies) / len(accuracies)

    return result


def create_bar_plot(category_accuracies, output_file):
    """Create a bar plot comparing model accuracies across categories."""
    # Add overall category
    category_accuracies = add_overall_accuracy(category_accuracies)

    # Get unique models and categories
    all_models = set()
    for category, models in category_accuracies.items():
        all_models.update(models.keys())

    all_models = sorted(list(all_models))

    # Sort categories but ensure 'Overall' is last
    categories = sorted([c for c in category_accuracies.keys() if c != "Overall"])
    if "Overall" in category_accuracies:
        categories.append("Overall")

    if not all_models or not categories:
        print("Error: No data to plot. Use --sample-data to generate sample data.")
        return False

    # Set up the plot
    plt.figure(figsize=(15, 10))

    # Set width of bars
    bar_width = 0.8 / len(all_models)

    # Set positions of bars on X axis
    r = np.arange(len(categories))

    # Create bars for each model
    for i, model in enumerate(all_models):
        accuracies = [
            category_accuracies[category].get(model, 0) for category in categories
        ]
        position = r + i * bar_width - (len(all_models) - 1) * bar_width / 2

        # Clean model name for legend by removing ":direct" suffix
        display_model = model
        if display_model.endswith(":direct"):
            display_model = display_model.replace(":direct", "")

        plt.bar(position, accuracies, width=bar_width, label=display_model)

    # Add labels and title
    plt.xlabel("Categories", fontweight="bold")
    plt.ylabel("Accuracy", fontweight="bold")
    plt.title("Model Accuracies by Category")

    # Add xticks on the middle of the group bars
    plt.xticks(r, categories, rotation=45, ha="right")

    # Create legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Bar plot saved to {output_file}")
    return True


def create_heatmap(category_accuracies, output_file):
    """Create a heatmap comparing model accuracies across categories."""
    # Add overall category
    category_accuracies = add_overall_accuracy(category_accuracies)

    # Get unique models and categories
    all_models = set()
    for category, models in category_accuracies.items():
        all_models.update(models.keys())

    # Clean model names for display
    all_models_display = {}
    for model in all_models:
        display_model = model
        if display_model.endswith(":direct"):
            display_model = display_model.replace(":direct", "")
        all_models_display[model] = display_model

    all_models = sorted(list(all_models))
    all_models_display_list = [all_models_display[model] for model in all_models]

    # Sort categories but ensure 'Overall' is last
    categories = sorted([c for c in category_accuracies.keys() if c != "Overall"])
    if "Overall" in category_accuracies:
        categories.append("Overall")

    if not all_models or not categories:
        print("Error: No data to plot. Use --sample-data to generate sample data.")
        return False

    # Create data matrix
    data = np.zeros((len(all_models), len(categories)))
    for i, model in enumerate(all_models):
        for j, category in enumerate(categories):
            data[i, j] = category_accuracies[category].get(model, 0)

    # Create heatmap
    plt.figure(figsize=(15, 10))
    plt.imshow(data, cmap="viridis", aspect="auto")

    # Add colorbar and labels
    plt.colorbar(label="Accuracy")
    plt.xlabel("Categories")
    plt.ylabel("Models")
    plt.title("Model Accuracies by Category")

    # Set tick labels
    plt.xticks(np.arange(len(categories)), categories, rotation=45, ha="right")
    plt.yticks(np.arange(len(all_models)), all_models_display_list)

    # Annotate cells with accuracy values
    for i in range(len(all_models)):
        for j in range(len(categories)):
            plt.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if data[i, j] < 0.7 else "black",
            )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Heatmap saved to {output_file}")
    return True


def main():
    args = parse_args()

    print(f"Analyzing MMLU-Pro results in {args.results_dir}...")
    category_accuracies = collect_model_accuracies(args.results_dir)

    # If no data found and sample-data flag is set, use sample data
    if (
        not category_accuracies
        or all(len(models) == 0 for models in category_accuracies.values())
    ) and args.sample_data:
        print("No results found. Using sample data for demonstration.")
        category_accuracies = generate_sample_data()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print(f"Generating {args.plot_type} plot...")
    success = False

    if args.plot_type == "bar":
        success = create_bar_plot(category_accuracies, args.output_file)
    else:  # heatmap
        success = create_heatmap(category_accuracies, args.output_file)

    if success:
        print("Done!")
    else:
        print("Failed to generate plot.")


if __name__ == "__main__":
    main()
