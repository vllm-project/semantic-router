import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evaluator import OnboardEvaluate
from .thresholds import build_threshold_report


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_config(config_path: Path, config: Dict[str, Any]) -> None:
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def _run_benchmark(args: argparse.Namespace) -> Optional[Path]:
    if not args.ml_benchmark_queries:
        return None

    queries = Path(args.ml_benchmark_queries)
    output = Path(args.ml_benchmark_output)
    bench_dir = Path(args.ml_selection_dir)
    cmd = [
        "python",
        str(bench_dir / "benchmark.py"),
        "--queries",
        str(queries),
        "--output",
        str(output),
    ]

    if args.ml_benchmark_model_config:
        cmd.extend(["--model-config", args.ml_benchmark_model_config])
    elif args.ml_benchmark_models:
        cmd.extend(["--models", args.ml_benchmark_models])
        if args.ml_benchmark_endpoint:
            cmd.extend(["--endpoint", args.ml_benchmark_endpoint])
    else:
        raise ValueError("Provide --ml-benchmark-models or --ml-benchmark-model-config")

    if args.ml_benchmark_concurrency:
        cmd.extend(["--concurrency", str(args.ml_benchmark_concurrency)])
    if args.ml_benchmark_max_tokens:
        cmd.extend(["--max-tokens", str(args.ml_benchmark_max_tokens)])
    if args.ml_benchmark_temperature is not None:
        cmd.extend(["--temperature", str(args.ml_benchmark_temperature)])
    if args.ml_benchmark_concise:
        cmd.append("--concise")
    if args.ml_benchmark_limit:
        cmd.extend(["--limit", str(args.ml_benchmark_limit)])

    subprocess.run(cmd, check=True)
    return output


def _run_training(args: argparse.Namespace, benchmark_output: Optional[Path]) -> None:
    if not args.ml_train_output:
        return

    train_dir = Path(args.ml_selection_dir)
    if not args.ml_train_data and benchmark_output is None:
        raise ValueError("Provide --ml-train-data or enable ML benchmark first")

    data_file = Path(args.ml_train_data or benchmark_output)

    cmd = [
        "python",
        str(train_dir / "train.py"),
        "--data-file",
        str(data_file),
        "--output-dir",
        str(args.ml_train_output),
    ]

    if args.ml_train_algorithm:
        cmd.extend(["--algorithm", args.ml_train_algorithm])
    if args.ml_train_device:
        cmd.extend(["--device", args.ml_train_device])
    if args.ml_train_embedding_model:
        cmd.extend(["--embedding-model", args.ml_train_embedding_model])
    if args.ml_train_quality_weight:
        cmd.extend(["--quality-weight", str(args.ml_train_quality_weight)])
    if args.ml_train_batch_size:
        cmd.extend(["--batch-size", str(args.ml_train_batch_size)])

    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model onboarding pipeline (eval, thresholds, ML routing models)"
    )
    parser.add_argument(
        "--config", required=True, help="Path to onboarding config JSON"
    )
    parser.add_argument(
        "--test-name",
        default="system_eval",
        choices=["system_eval", "arc_challenge", "mmlu_pro"],
        help="Evaluation to run",
    )
    parser.add_argument("--datasets", nargs="*", help="System eval dataset IDs")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--input-path", help="Text/jsonl input for system_eval")
    parser.add_argument("--report-out", help="Write JSON report to this path")

    parser.add_argument("--thresholds-out", help="Write threshold report JSON")
    parser.add_argument("--min-accuracy", type=float, default=0.7)
    parser.add_argument("--max-latency-ms", type=float, default=2000.0)
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Write onboarding thresholds back into config JSON",
    )
    parser.add_argument("--config-out", help="Write updated config to a new path")

    parser.add_argument(
        "--ml-selection-dir",
        default="src/training/model_selection/ml_model_selection",
        help="Path to ML model selection directory",
    )
    parser.add_argument("--ml-benchmark-queries", help="Queries JSONL for ML benchmark")
    parser.add_argument("--ml-benchmark-models", help="Comma-separated model list")
    parser.add_argument("--ml-benchmark-model-config", help="models.yaml for benchmark")
    parser.add_argument("--ml-benchmark-endpoint", help="Endpoint for --models mode")
    parser.add_argument("--ml-benchmark-output", default="benchmark_output.jsonl")
    parser.add_argument("--ml-benchmark-concurrency", type=int)
    parser.add_argument("--ml-benchmark-max-tokens", type=int)
    parser.add_argument("--ml-benchmark-temperature", type=float)
    parser.add_argument("--ml-benchmark-concise", action="store_true")
    parser.add_argument("--ml-benchmark-limit", type=int)

    parser.add_argument("--ml-train-data", help="Benchmark output JSONL for training")
    parser.add_argument("--ml-train-output", help="Output dir for trained models")
    parser.add_argument("--ml-train-algorithm", help="all|knn|kmeans|svm|mlp")
    parser.add_argument("--ml-train-device", help="cpu|cuda|mps")
    parser.add_argument("--ml-train-embedding-model", help="Embedding model name")
    parser.add_argument("--ml-train-quality-weight", type=float)
    parser.add_argument("--ml-train-batch-size", type=int)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)

    evaluator = OnboardEvaluate(config_path=str(config_path))
    evaluator.parse(config)

    if args.test_name == "system_eval":
        datasets = args.datasets
        result = evaluator.run_performance_test(
            "system_eval",
            datasets=datasets,
            max_samples=args.max_samples,
            input_path=args.input_path,
        )
    else:
        result = evaluator.run_performance_test(args.test_name)

    if args.report_out:
        evaluator.generate_report(args.report_out)

    threshold_report = None
    if args.test_name == "system_eval" and evaluator.system_eval_summary:
        threshold_report = build_threshold_report(
            evaluator.system_eval_summary,
            args.min_accuracy,
            args.max_latency_ms,
        )
        if args.thresholds_out:
            threshold_path = Path(args.thresholds_out)
            with threshold_path.open("w", encoding="utf-8") as handle:
                json.dump(threshold_report, handle, indent=2)
                handle.write("\n")

    if args.update_config and threshold_report is not None:
        config["onboarding_thresholds"] = threshold_report
        output_path = Path(args.config_out) if args.config_out else config_path
        _write_config(output_path, config)

    benchmark_output = _run_benchmark(args)
    _run_training(args, benchmark_output)

    print(
        f"Completed {result.test_name} with score {result.score:.4f} (samples={result.metrics.get('total_samples')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
