import argparse
import os


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for reasoning mode evaluation."""
    parser = argparse.ArgumentParser(
        description="Reasoning Mode Evaluation Benchmark (Issue #42)"
    )
    _add_dataset_args(parser)
    _add_endpoint_args(parser)
    _add_execution_args(parser)
    _add_output_args(parser)
    return parser.parse_args()


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["mmlu", "gpqa"],
        help="Datasets to evaluate (supports MMLU and non-MMLU). Use --list-datasets to see all.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=10,
        help="Number of questions to sample per category",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to evaluate",
    )


def _add_endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"),
        help="vLLM endpoint URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get(
            "VLLM_API_KEY", os.environ.get("OPENAI_API_KEY", "1234")
        ),
        help="API key for endpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to evaluate (if not specified, fetches from endpoint)",
    )
    parser.add_argument(
        "--reasoning-family",
        type=str,
        default=None,
        help="Reasoning family identifier (for example qwen3, deepseek, or gpt-oss).",
    )


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: dataset-optimal)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/reasoning_mode_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate markdown report",
    )
