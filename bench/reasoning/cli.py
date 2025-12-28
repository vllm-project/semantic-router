#!/usr/bin/env python3
"""
Command Line Interface for Semantic Router Benchmark Suite
"""

import argparse
import os
import sys
from typing import List, Optional


def main():
    """Main CLI entry point for semantic-router-bench."""
    parser = argparse.ArgumentParser(
        prog="semantic-router-bench",
        description="Comprehensive benchmark suite for semantic router vs direct vLLM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick dataset test
  semantic-router-bench test --dataset mmlu --samples 5

  # Full benchmark comparison
  semantic-router-bench compare --dataset arc-challenge --samples 10

  # Reasoning mode evaluation (Issue #42)
  semantic-router-bench reasoning-eval --datasets mmlu gpqa --samples 10

  # List available datasets
  semantic-router-bench list-datasets

  # Generate plots from existing results
  semantic-router-bench plot --router-dir results/router_mmlu --vllm-dir results/vllm_mmlu

For more detailed usage, see: https://vllm-semantic-router.com/docs/benchmarking
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test command - quick single dataset evaluation
    test_parser = subparsers.add_parser("test", help="Quick test on a single dataset")
    test_parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "mmlu",
            "arc",
            "arc-challenge",
            "gpqa",
            "truthfulqa",
            "commonsenseqa",
            "hellaswag",
        ],
        help="Dataset to test",
    )
    test_parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples per category (default: 5)",
    )
    test_parser.add_argument(
        "--mode",
        choices=["router", "vllm", "both"],
        default="both",
        help="Evaluation mode (default: both)",
    )
    test_parser.add_argument(
        "--output-dir",
        default="results/quick_test",
        help="Output directory for results",
    )

    # Compare command - full router vs vLLM comparison
    compare_parser = subparsers.add_parser(
        "compare", help="Full router vs vLLM comparison"
    )
    compare_parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "mmlu",
            "arc",
            "arc-challenge",
            "gpqa",
            "truthfulqa",
            "commonsenseqa",
            "hellaswag",
        ],
        help="Dataset to benchmark",
    )
    compare_parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per category (default: 10)",
    )
    compare_parser.add_argument(
        "--router-endpoint",
        default="http://127.0.0.1:8801/v1",
        help="Router endpoint URL",
    )
    compare_parser.add_argument(
        "--vllm-endpoint", default="http://127.0.0.1:8000/v1", help="vLLM endpoint URL"
    )
    compare_parser.add_argument(
        "--vllm-model", default="openai/gpt-oss-20b", help="vLLM model name"
    )
    compare_parser.add_argument(
        "--output-dir",
        default="results/comparison",
        help="Output directory for results",
    )

    # List datasets command
    list_parser = subparsers.add_parser("list-datasets", help="List available datasets")

    # Plot command - generate plots from existing results
    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from benchmark results"
    )
    plot_parser.add_argument(
        "--router-dir", required=True, help="Directory containing router results"
    )
    plot_parser.add_argument(
        "--vllm-dir", required=True, help="Directory containing vLLM results"
    )
    plot_parser.add_argument(
        "--output-dir", default="results/plots", help="Output directory for plots"
    )
    plot_parser.add_argument("--dataset-name", help="Dataset name for plot titles")

    # Comprehensive command - run full research benchmark
    comprehensive_parser = subparsers.add_parser(
        "comprehensive", help="Run comprehensive multi-dataset benchmark"
    )
    comprehensive_parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "mmlu",
            "arc-challenge",
            "gpqa",
            "truthfulqa",
            "commonsenseqa",
            "hellaswag",
        ],
        help="Datasets to benchmark",
    )
    comprehensive_parser.add_argument(
        "--router-endpoint", default="http://127.0.0.1:8801/v1"
    )
    comprehensive_parser.add_argument(
        "--vllm-endpoint", default="http://127.0.0.1:8000/v1"
    )
    comprehensive_parser.add_argument("--vllm-model", default="openai/gpt-oss-20b")

    # Reasoning mode evaluation command (Issue #42)
    reasoning_parser = subparsers.add_parser(
        "reasoning-eval",
        help="Evaluate standard vs reasoning mode (Issue #42)",
    )
    reasoning_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mmlu", "gpqa"],
        help="Datasets to evaluate (MMLU and non-MMLU)",
    )
    reasoning_parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per category (default: 10)",
    )
    reasoning_parser.add_argument(
        "--endpoint",
        default=os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"),
        help="vLLM endpoint URL",
    )
    reasoning_parser.add_argument(
        "--model",
        default=None,
        help="Model to evaluate (fetches from endpoint if not specified)",
    )
    reasoning_parser.add_argument(
        "--output-dir",
        default="results/reasoning_mode_eval",
        help="Output directory for results",
    )
    reasoning_parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests",
    )
    reasoning_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    reasoning_parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip markdown report generation",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Import modules only when needed to speed up CLI startup
    if args.command == "test":
        return run_test(args)
    elif args.command == "compare":
        return run_compare(args)
    elif args.command == "list-datasets":
        return list_datasets()
    elif args.command == "plot":
        return run_plot(args)
    elif args.command == "comprehensive":
        return run_comprehensive(args)
    elif args.command == "reasoning-eval":
        return run_reasoning_eval(args)
    else:
        parser.print_help()
        return 1


def run_test(args):
    """Run quick test command."""
    print(f"🧪 Quick test: {args.dataset} dataset ({args.samples} samples)")

    # Import and run the benchmark script
    import os
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "reasoning.router_reason_bench_multi_dataset",
        "--dataset",
        args.dataset,
        "--samples-per-category",
        str(args.samples),
        "--output-dir",
        args.output_dir,
        "--seed",
        "42",
    ]

    if args.mode in ["router", "both"]:
        cmd.extend(["--run-router", "--router-models", "auto"])

    if args.mode in ["vllm", "both"]:
        cmd.extend(
            [
                "--run-vllm",
                "--vllm-models",
                "openai/gpt-oss-20b",
                "--vllm-exec-modes",
                "NR",
                "NR_REASONING",
            ]
        )

    return subprocess.call(cmd)


def run_compare(args):
    """Run comparison command."""
    print(f"⚡ Comparison: {args.dataset} dataset ({args.samples} samples)")

    import os
    import subprocess

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "benchmark_comparison.sh"
    )
    cmd = [script_path, args.dataset, str(args.samples)]

    env = os.environ.copy()
    env.update(
        {
            "ROUTER_ENDPOINT": args.router_endpoint,
            "VLLM_ENDPOINT": args.vllm_endpoint,
            "VLLM_MODEL": args.vllm_model,
            "OUTPUT_DIR": args.output_dir,
        }
    )

    return subprocess.call(cmd, env=env)


def list_datasets():
    """List available datasets."""
    try:
        from .dataset_factory import list_available_datasets

        # This function prints the datasets and returns None
        list_available_datasets()

        print("\nUsage examples:")
        print("  semantic-router-bench test --dataset mmlu --samples 5")
        print("  semantic-router-bench compare --dataset arc-challenge --samples 10")

        return 0
    except ImportError as e:
        print(f"Error importing dataset factory: {e}")
        return 1


def run_plot(args):
    """Run plotting command."""
    print(f"📈 Generating plots from {args.router_dir} and {args.vllm_dir}")

    import os
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "reasoning.bench_plot",
        "--router-dir",
        args.router_dir,
        "--vllm-dir",
        args.vllm_dir,
        "--output-dir",
        args.output_dir,
    ]

    if args.dataset_name:
        cmd.extend(["--dataset-name", args.dataset_name])

    return subprocess.call(cmd)


def run_comprehensive(args):
    """Run comprehensive benchmark."""
    print(f"🔬 Comprehensive benchmark: {', '.join(args.datasets)}")

    import os
    import subprocess

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "comprehensive_bench.sh"
    )

    env = os.environ.copy()
    env.update(
        {
            "ROUTER_ENDPOINT": args.router_endpoint,
            "VLLM_ENDPOINT": args.vllm_endpoint,
            "VLLM_MODEL": args.vllm_model,
            "DATASETS": " ".join(args.datasets),
        }
    )

    return subprocess.call([script_path], env=env)


def run_reasoning_eval(args):
    """Run reasoning mode evaluation (Issue #42).

    Compares standard vs reasoning mode with metrics:
    - Response correctness on MMLU(-Pro) and non-MMLU test sets
    - Token usage (completion_tokens/prompt_tokens ratio)
    - Response time per output token
    """
    print(f"🧠 Reasoning Mode Evaluation (Issue #42)")
    print(f"   Datasets: {', '.join(args.datasets)}")
    print(f"   Samples per category: {args.samples}")

    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "reasoning.reasoning_mode_eval",
        "--datasets",
        *args.datasets,
        "--samples-per-category",
        str(args.samples),
        "--endpoint",
        args.endpoint,
        "--output-dir",
        args.output_dir,
        "--concurrent-requests",
        str(args.concurrent),
    ]

    if args.model:
        cmd.extend(["--model", args.model])

    if args.no_plots:
        cmd.append("--generate-plots")
        cmd.append("False")

    if args.no_report:
        cmd.append("--generate-report")
        cmd.append("False")

    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
