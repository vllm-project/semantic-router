#!/usr/bin/env python3
"""
Comprehensive Results Plotting Script

This script creates comparison plots showing:
1. Accuracy comparison across datasets and modes
2. Token usage comparison across datasets and modes

Modes compared:
- Router (auto model with reasoning)
- vLLM Direct (No Reasoning)
- vLLM Direct (All Reasoning)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_and_clean_data(csv_path):
    """Load and clean the research results CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} records from {csv_path}")

        # Show available modes
        print(f"📊 Available modes: {df['Mode'].unique().tolist()}")
        print(f"📊 Available datasets: {df['Dataset'].unique().tolist()}")
        print(f"📊 Available models: {df['Model'].unique().tolist()}")

        # Map old mode names to new descriptive names for backward compatibility
        mode_mapping = {
            "vLLM_NR": "vLLM_No_Reasoning",
            "vLLM_XC": "vLLM_All_Reasoning",
            "vLLM_NR_REASONING": "vLLM_All_Reasoning",
        }
        df["Mode"] = df["Mode"].replace(mode_mapping)

        # Clean data
        df = df.dropna(subset=["Accuracy", "Avg_Total_Tokens"])
        df = df[df["Accuracy"] >= 0]  # Remove invalid accuracy values

        # Get the latest results for each dataset/mode/model combination
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df_latest = (
            df.sort_values("Timestamp").groupby(["Dataset", "Mode", "Model"]).tail(1)
        )

        print(f"✅ Using {len(df_latest)} latest records after cleaning")
        return df_latest

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def create_accuracy_plot(df, output_dir):
    """Create accuracy comparison plot."""
    plt.figure(figsize=(14, 8))

    # Prepare data for plotting
    datasets = sorted(df["Dataset"].unique())
    modes = ["Router", "vLLM_No_Reasoning", "vLLM_All_Reasoning"]

    # Filter for available modes
    available_modes = [mode for mode in modes if mode in df["Mode"].unique()]

    # Create subplot data
    x = np.arange(len(datasets))
    width = 0.25

    # Colors for each mode
    colors = {
        "Router": "#2E86AB",
        "vLLM_No_Reasoning": "#A23B72",
        "vLLM_All_Reasoning": "#F18F01",
    }

    # Plot bars for each mode
    for i, mode in enumerate(available_modes):
        mode_data = df[df["Mode"] == mode]
        accuracies = []

        for dataset in datasets:
            dataset_data = mode_data[mode_data["Dataset"] == dataset]
            if not dataset_data.empty:
                # Use the latest model's accuracy
                accuracy = dataset_data.iloc[-1]["Accuracy"]
                accuracies.append(accuracy)
            else:
                accuracies.append(0)

        # Clean mode name for display
        display_name = mode.replace("vLLM_", "vLLM ").replace("_", " ")

        plt.bar(
            x + i * width,
            accuracies,
            width,
            label=display_name,
            color=colors.get(mode, f"C{i}"),
            alpha=0.8,
        )

        # Add value labels on bars
        for j, acc in enumerate(accuracies):
            if acc > 0:
                plt.text(
                    x[j] + i * width,
                    acc + 0.01,
                    f"{acc:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.xlabel("Dataset", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
    plt.title(
        "Accuracy Comparison: Router vs vLLM Direct\n(No Reasoning vs All Reasoning)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(x + width, [d.upper() for d in datasets], rotation=45, ha="right")
    plt.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylim(0, 1.1)

    # Add model info
    models = df["Model"].unique()
    model_text = f"Models: {', '.join(models)}"
    plt.figtext(0.02, 0.02, model_text, fontsize=8, style="italic")

    plt.tight_layout()

    # Save plot
    accuracy_path = output_dir / "accuracy_comparison.png"
    plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
    print(f"📊 Accuracy plot saved: {accuracy_path}")
    plt.close()


def create_token_usage_plot(df, output_dir):
    """Create token usage comparison plot."""
    plt.figure(figsize=(14, 8))

    # Prepare data for plotting
    datasets = sorted(df["Dataset"].unique())
    modes = ["Router", "vLLM_No_Reasoning", "vLLM_All_Reasoning"]

    # Filter for available modes
    available_modes = [mode for mode in modes if mode in df["Mode"].unique()]

    # Create subplot data
    x = np.arange(len(datasets))
    width = 0.25

    # Colors for each mode
    colors = {
        "Router": "#2E86AB",
        "vLLM_No_Reasoning": "#A23B72",
        "vLLM_All_Reasoning": "#F18F01",
    }

    # Plot bars for each mode
    for i, mode in enumerate(available_modes):
        mode_data = df[df["Mode"] == mode]
        token_usage = []

        for dataset in datasets:
            dataset_data = mode_data[mode_data["Dataset"] == dataset]
            if not dataset_data.empty:
                # Use the latest model's token usage
                tokens = dataset_data.iloc[-1]["Avg_Total_Tokens"]
                token_usage.append(tokens)
            else:
                token_usage.append(0)

        # Clean mode name for display
        display_name = mode.replace("vLLM_", "vLLM ").replace("_", " ")

        plt.bar(
            x + i * width,
            token_usage,
            width,
            label=display_name,
            color=colors.get(mode, f"C{i}"),
            alpha=0.8,
        )

        # Add value labels on bars
        for j, tokens in enumerate(token_usage):
            if tokens > 0:
                plt.text(
                    x[j] + i * width,
                    tokens + max(token_usage) * 0.01,
                    f"{tokens:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.xlabel("Dataset", fontsize=12, fontweight="bold")
    plt.ylabel("Average Total Tokens", fontsize=12, fontweight="bold")
    plt.title(
        "Token Usage Comparison: Router vs vLLM Direct\n(No Reasoning vs All Reasoning)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(x + width, [d.upper() for d in datasets], rotation=45, ha="right")
    plt.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, axis="y")

    # Add model info
    models = df["Model"].unique()
    model_text = f"Models: {', '.join(models)}"
    plt.figtext(0.02, 0.02, model_text, fontsize=8, style="italic")

    plt.tight_layout()

    # Save plot
    token_path = output_dir / "token_usage_comparison.png"
    plt.savefig(token_path, dpi=300, bbox_inches="tight")
    print(f"📊 Token usage plot saved: {token_path}")
    plt.close()


def create_efficiency_plot(df, output_dir):
    """Create efficiency scatter plot (accuracy vs tokens)."""
    plt.figure(figsize=(12, 8))

    modes = ["Router", "vLLM_No_Reasoning", "vLLM_All_Reasoning"]
    available_modes = [mode for mode in modes if mode in df["Mode"].unique()]

    colors = {
        "Router": "#2E86AB",
        "vLLM_No_Reasoning": "#A23B72",
        "vLLM_All_Reasoning": "#F18F01",
    }

    markers = {"Router": "o", "vLLM_No_Reasoning": "s", "vLLM_All_Reasoning": "^"}

    for mode in available_modes:
        mode_data = df[df["Mode"] == mode]

        display_name = mode.replace("vLLM_", "vLLM ").replace("_", " ")

        plt.scatter(
            mode_data["Avg_Total_Tokens"],
            mode_data["Accuracy"],
            c=colors.get(mode, "gray"),
            marker=markers.get(mode, "o"),
            s=100,
            alpha=0.7,
            label=display_name,
            edgecolors="black",
            linewidth=1,
        )

        # Add dataset labels
        for _, row in mode_data.iterrows():
            plt.annotate(
                row["Dataset"].upper(),
                (row["Avg_Total_Tokens"], row["Accuracy"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

    plt.xlabel("Average Total Tokens", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
    plt.title(
        "Efficiency Analysis: Accuracy vs Token Usage\n(Higher accuracy with lower tokens is better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)

    # Add model info
    models = df["Model"].unique()
    model_text = f"Models: {', '.join(models)}"
    plt.figtext(0.02, 0.02, model_text, fontsize=8, style="italic")

    plt.tight_layout()

    # Save plot
    efficiency_path = output_dir / "efficiency_analysis.png"
    plt.savefig(efficiency_path, dpi=300, bbox_inches="tight")
    print(f"📊 Efficiency plot saved: {efficiency_path}")
    plt.close()


def create_summary_table(df, output_dir):
    """Create a summary table of results."""

    # Check if we're dealing with a single dataset
    unique_datasets = df["Dataset"].nunique()
    dataset_name = (
        df["Dataset"].iloc[0] if unique_datasets == 1 else "Multiple Datasets"
    )

    if unique_datasets == 1:
        print(f"\n📋 RESULTS SUMMARY - {dataset_name.upper()}")
    else:
        print(f"\n📋 RESULTS SUMMARY - AGGREGATE ACROSS {unique_datasets} DATASETS")
    print("=" * 80)

    # For single dataset, show individual values; for multiple datasets, show statistics
    if unique_datasets == 1:
        # Show individual values for each mode
        print(
            f"{'Mode':<20} {'Accuracy':<10} {'Tokens':<10} {'Latency(ms)':<12} {'Samples':<8}"
        )
        print("-" * 65)

        for mode in sorted(df["Mode"].unique()):
            mode_data = df[df["Mode"] == mode].iloc[0]
            print(
                f"{mode:<20} {mode_data['Accuracy']:<10.3f} {mode_data['Avg_Total_Tokens']:<10.1f} {mode_data['Avg_Latency_ms']:<12.1f} {mode_data['Sample_Count']:<8}"
            )

        summary = df.groupby("Mode")[
            ["Accuracy", "Avg_Total_Tokens", "Avg_Latency_ms"]
        ].first()
    else:
        # Group by mode and calculate averages for multiple datasets
        summary = (
            df.groupby("Mode")
            .agg(
                {
                    "Accuracy": ["mean", "std", "count"],
                    "Avg_Total_Tokens": ["mean", "std"],
                    "Avg_Latency_ms": ["mean", "std"],
                }
            )
            .round(3)
        )

        print(summary)

    # Save detailed results table
    detailed_table = df.pivot_table(
        index="Dataset",
        columns="Mode",
        values=["Accuracy", "Avg_Total_Tokens"],
        aggfunc="mean",
    ).round(3)

    table_path = output_dir / "results_summary_table.csv"
    detailed_table.to_csv(table_path)
    print(f"\n📊 Detailed results table saved: {table_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot comprehensive benchmark results")
    parser.add_argument(
        "--csv",
        type=str,
        default="research_results_master.csv",
        help="Path to research results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help='Filter results for specific model (e.g., "Qwen/Qwen3-30B-A3B")',
    )
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help='Filter results for specific dataset (e.g., "truthfulqa")',
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"🎨 Creating comprehensive benchmark plots...")
    print(f"📊 Input CSV: {args.csv}")
    print(f"📁 Output directory: {output_dir}")

    # Load data
    df = load_and_clean_data(args.csv)

    # Filter by model if specified
    if args.model_filter:
        df = df[df["Model"].str.contains(args.model_filter, na=False)]
        print(f"🔍 Filtered to model: {args.model_filter} ({len(df)} records)")

    # Filter by dataset if specified
    if args.dataset_filter:
        df = df[df["Dataset"].str.contains(args.dataset_filter, case=False, na=False)]
        print(f"🔍 Filtered to dataset: {args.dataset_filter} ({len(df)} records)")

    if df.empty:
        print("❌ No data available after filtering!")
        sys.exit(1)

    # Create plots
    create_accuracy_plot(df, output_dir)
    create_token_usage_plot(df, output_dir)
    create_efficiency_plot(df, output_dir)

    # Create summary
    summary = create_summary_table(df, output_dir)

    print(f"\n🎉 All plots created successfully!")
    print(f"📁 Check the '{output_dir}' directory for:")
    print(f"   - accuracy_comparison.png")
    print(f"   - token_usage_comparison.png")
    print(f"   - efficiency_analysis.png")
    print(f"   - results_summary_table.csv")


if __name__ == "__main__":
    main()
