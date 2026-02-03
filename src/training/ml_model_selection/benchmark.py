#!/usr/bin/env python3
"""
Benchmark script for ML model selection training data generation.

This script:
1. Takes input queries (JSONL format with 'query' and optionally 'ground_truth')
2. Runs each query against multiple LLM endpoints
3. Measures performance (accuracy) and response_time
4. Outputs benchmark_training_data.jsonl for training

Usage:
    # Simple: All models on same endpoint (e.g., vLLM serving multiple models)
    python benchmark.py --queries queries.jsonl --models llama-3.2-1b,mistral-7b

    # Different endpoints/auth per model: Use config file
    python benchmark.py --queries queries.jsonl --model-config models.yaml

    # Output to specific file
    python benchmark.py --queries queries.jsonl --models llama-3.2-1b --output my_benchmark.jsonl

Config file format (models.yaml):
    models:
      - name: llama-3.2-1b
        endpoint: http://localhost:8000/v1
        # No auth needed for local model

      - name: gpt-4
        endpoint: https://api.openai.com/v1
        api_key: ${OPENAI_API_KEY}  # Environment variable

      - name: claude-3
        endpoint: https://api.anthropic.com/v1
        api_key: sk-ant-xxx  # Direct value

      - name: custom-model
        endpoint: https://custom.api.com/v1
        headers:
          Authorization: Bearer ${CUSTOM_TOKEN}
          X-Custom-Header: value

After benchmarking, run add_category_to_training_data.py to add category field:
    python ../../../src/semantic-router/pkg/modelselection/data/add_category_to_training_data.py \\
        --input my_benchmark.jsonl --output my_benchmark_with_category.jsonl

Then train with:
    python train.py --data-file my_benchmark_with_category.jsonl --output-dir models/
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library required. Install with: pip install openai")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None  # Will error if config file is used

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    endpoint: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    max_tokens: int = 1024
    temperature: float = 0.0

    def get_client(self) -> OpenAI:
        """Create OpenAI client for this model."""
        # Resolve environment variables in api_key
        api_key = self.api_key or "dummy"
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "dummy")

        # Resolve environment variables in headers
        default_headers = None
        if self.headers:
            default_headers = {}
            for key, value in self.headers.items():
                if (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    env_var = value[2:-1]
                    value = os.environ.get(env_var, "")
                default_headers[key] = value

        return OpenAI(
            base_url=self.endpoint,
            api_key=api_key,
            default_headers=default_headers,
        )


def load_model_configs(config_path: Path) -> List[ModelConfig]:
    """Load model configurations from YAML file."""
    if yaml is None:
        print(
            "Error: PyYAML required for config files. Install with: pip install pyyaml"
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    configs = []
    for model_data in data.get("models", []):
        config = ModelConfig(
            name=model_data["name"],
            endpoint=model_data.get("endpoint", "http://localhost:8000/v1"),
            api_key=model_data.get("api_key"),
            headers=model_data.get("headers"),
            max_tokens=model_data.get("max_tokens", 1024),
            temperature=model_data.get("temperature", 0.0),
        )
        configs.append(config)

    return configs


def create_model_configs_from_list(
    models: List[str],
    endpoint: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> List[ModelConfig]:
    """Create model configs from simple comma-separated list."""
    return [
        ModelConfig(
            name=model,
            endpoint=endpoint,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for model in models
    ]


@dataclass
class QueryRecord:
    """A single query record for benchmarking."""

    query: str
    ground_truth: Optional[str] = None
    task_name: Optional[str] = None
    metric: Optional[str] = None
    embedding_id: Optional[int] = None
    choices: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single query against a model."""

    query: str
    model_name: str
    response: str
    performance: float  # 0.0 - 1.0
    response_time: float  # seconds
    ground_truth: Optional[str] = None
    task_name: Optional[str] = None
    metric: Optional[str] = None
    embedding_id: Optional[int] = None
    choices: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """Convert to JSONL-compatible dict (same format as benchmark_training_data.jsonl)."""
        result = {
            "query": self.query,
            "model_name": self.model_name,
            "response": self.response,
            "performance": self.performance,
            "response_time": self.response_time,
        }

        # Add optional fields
        if self.ground_truth is not None:
            result["ground_truth"] = self.ground_truth
        if self.task_name is not None:
            result["task_name"] = self.task_name
        if self.metric is not None:
            result["metric"] = self.metric
        if self.embedding_id is not None:
            result["embedding_id"] = self.embedding_id
        if self.choices is not None:
            result["choices"] = self.choices

        # Add any extra fields from input
        result.update(self.extra_fields)

        return result


def load_queries(file_path: Path) -> List[QueryRecord]:
    """Load queries from JSONL file."""
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                query = data.get("query", "")
                if not query:
                    print(f"Warning: Skipping line {line_num} - no query field")
                    continue

                # Extract known fields
                record = QueryRecord(
                    query=query,
                    ground_truth=data.get("ground_truth"),
                    task_name=data.get("task_name"),
                    metric=data.get("metric"),
                    embedding_id=data.get("embedding_id"),
                    choices=data.get("choices"),
                )

                # Store any extra fields
                known_fields = {
                    "query",
                    "ground_truth",
                    "task_name",
                    "metric",
                    "embedding_id",
                    "choices",
                    "model_name",
                    "response",
                    "performance",
                    "response_time",
                    "category",
                }
                for key, value in data.items():
                    if key not in known_fields:
                        record.extra_fields[key] = value

                records.append(record)

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")

    print(f"Loaded {len(records)} queries from {file_path}")
    return records


def evaluate_response(
    response: str,
    ground_truth: Optional[str],
    metric: Optional[str] = None,
    choices: Optional[str] = None,
) -> float:
    """
    Evaluate response against ground truth.

    Returns:
        Performance score between 0.0 and 1.0
    """
    if ground_truth is None:
        # No ground truth - return 0.5 as neutral score
        return 0.5

    response_lower = response.lower().strip()
    truth_lower = ground_truth.lower().strip()

    # Exact match
    if response_lower == truth_lower:
        return 1.0

    # Check if ground truth is in response
    if truth_lower in response_lower:
        return 0.9

    # For multiple choice questions
    if choices:
        # Extract answer letter from response (e.g., "A", "B", "C", "D")
        answer_pattern = re.compile(r"(?:answer(?:\s*is)?:?\s*)([A-J])", re.IGNORECASE)
        match = answer_pattern.search(response)
        if match:
            predicted = match.group(1).upper()
            # Ground truth might be just the letter
            if predicted == truth_lower.upper():
                return 1.0
            # Or it might be the full answer text
            if predicted in truth_lower.upper():
                return 1.0

    # For boxed math answers
    boxed_pattern = re.compile(r"\\boxed\{([^}]+)\}")
    response_boxed = boxed_pattern.search(response)
    truth_boxed = boxed_pattern.search(ground_truth)

    if response_boxed and truth_boxed:
        if response_boxed.group(1).strip() == truth_boxed.group(1).strip():
            return 1.0
    elif truth_boxed:
        # Ground truth has boxed, check if response contains the answer
        truth_answer = truth_boxed.group(1).strip()
        if truth_answer.lower() in response_lower:
            return 0.8

    # Partial match based on word overlap
    response_words = set(response_lower.split())
    truth_words = set(truth_lower.split())

    if truth_words:
        overlap = len(response_words & truth_words) / len(truth_words)
        return min(overlap, 0.7)  # Cap at 0.7 for partial matches

    return 0.0


def benchmark_query(
    model_config: ModelConfig,
    query: QueryRecord,
) -> BenchmarkResult:
    """Benchmark a single query against a model."""

    client = model_config.get_client()
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_config.name,
            messages=[{"role": "user", "content": query.query}],
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )

        response_text = response.choices[0].message.content or ""
        success = True

    except Exception as e:
        response_text = f"Error: {str(e)}"
        success = False

    end_time = time.time()
    response_time = end_time - start_time

    # Evaluate performance
    if success:
        performance = evaluate_response(
            response_text,
            query.ground_truth,
            query.metric,
            query.choices,
        )
    else:
        performance = 0.0

    return BenchmarkResult(
        query=query.query,
        model_name=model_config.name,
        response=response_text,
        performance=performance,
        response_time=response_time,
        ground_truth=query.ground_truth,
        task_name=query.task_name,
        metric=query.metric,
        embedding_id=query.embedding_id,
        choices=query.choices,
        extra_fields=query.extra_fields,
    )


def run_benchmark(
    queries: List[QueryRecord],
    model_configs: List[ModelConfig],
    concurrency: int = 4,
    progress: bool = True,
) -> List[BenchmarkResult]:
    """Run benchmark for all queries against all models."""

    results = []

    # Create tasks: (query, model_config) pairs
    tasks = [(q, m) for q in queries for m in model_configs]
    total_tasks = len(tasks)

    # Group models by endpoint for display
    endpoints = set(m.endpoint for m in model_configs)
    model_names = [m.name for m in model_configs]

    print(
        f"\nBenchmarking {len(queries)} queries × {len(model_configs)} models = {total_tasks} requests"
    )
    print(f"Models: {', '.join(model_names)}")
    if len(endpoints) == 1:
        print(f"Endpoint: {list(endpoints)[0]}")
    else:
        print(f"Endpoints: {len(endpoints)} different endpoints")
        for m in model_configs:
            auth_info = "with API key" if m.api_key else "no auth"
            print(f"  - {m.name}: {m.endpoint} ({auth_info})")
    print(f"Concurrency: {concurrency}")
    print()

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}

        for query, model_config in tasks:
            future = executor.submit(
                benchmark_query,
                model_config,
                query,
            )
            futures[future] = (query, model_config)

        # Process results as they complete
        iterator = as_completed(futures)
        if progress:
            iterator = tqdm(iterator, total=total_tasks, desc="Benchmarking")

        for future in iterator:
            query, model_config = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if result.performance == 0.0 and "Error" in result.response:
                    failed += 1

            except Exception as e:
                print(f"\nError processing {model_config.name}: {e}")
                failed += 1

    print(f"\nCompleted: {completed}/{total_tasks} ({failed} errors)")

    return results


def save_results(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSONL file."""

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_jsonl_dict()) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print benchmark summary."""

    print("\n" + "=" * 60)
    print("  Benchmark Summary")
    print("=" * 60)

    # Group by model
    model_stats: Dict[str, Dict[str, Any]] = {}

    for result in results:
        if result.model_name not in model_stats:
            model_stats[result.model_name] = {
                "count": 0,
                "total_perf": 0.0,
                "total_time": 0.0,
                "successes": 0,
            }

        stats = model_stats[result.model_name]
        stats["count"] += 1
        stats["total_perf"] += result.performance
        stats["total_time"] += result.response_time

        if result.performance > 0:
            stats["successes"] += 1

    print(
        f"\n{'Model':<25} {'Queries':>8} {'Avg Perf':>10} {'Avg Time':>10} {'Success':>10}"
    )
    print("-" * 65)

    for model, stats in sorted(model_stats.items()):
        avg_perf = stats["total_perf"] / stats["count"] if stats["count"] > 0 else 0
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        success_rate = (
            stats["successes"] / stats["count"] * 100 if stats["count"] > 0 else 0
        )

        print(
            f"{model:<25} {stats['count']:>8} {avg_perf:>10.3f} {avg_time:>9.2f}s {success_rate:>9.1f}%"
        )

    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs for ML model selection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple: All models on same endpoint (local vLLM)
  python benchmark.py --queries queries.jsonl --models llama-3.2-1b,mistral-7b,codellama-7b

  # With custom endpoint and API key
  python benchmark.py --queries queries.jsonl --models gpt-4 \\
      --endpoint https://api.openai.com/v1 --api-key $OPENAI_API_KEY

  # Mixed models with config file (different endpoints/auth per model)
  python benchmark.py --queries queries.jsonl --model-config models.yaml

  # High concurrency for faster benchmarking
  python benchmark.py --queries queries.jsonl --models model1,model2 --concurrency 16

Config file format (models.yaml):
  models:
    - name: llama-3.2-1b           # Local model, no auth
      endpoint: http://localhost:8000/v1

    - name: gpt-4                   # OpenAI with API key
      endpoint: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}

    - name: claude-3                # Anthropic with direct key
      endpoint: https://api.anthropic.com/v1
      api_key: sk-ant-xxx

    - name: custom-model            # Custom headers
      endpoint: https://custom.api.com/v1
      headers:
        Authorization: Bearer ${CUSTOM_TOKEN}

After benchmarking:
  # Add categories using VSR classifier
  python add_category_to_training_data.py --input benchmark_output.jsonl --output benchmark_with_category.jsonl

  # Train models
  python train.py --data-file benchmark_with_category.jsonl --output-dir models/
        """,
    )

    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to JSONL file with queries (must have 'query' field, optionally 'ground_truth')",
    )

    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names (all use same endpoint)",
    )
    model_group.add_argument(
        "--model-config",
        type=str,
        help="Path to YAML config file with model definitions (supports different endpoints/auth per model)",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1"),
        help="API endpoint for --models (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get(
            "LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy")
        ),
        help="API key for --models (uses LLM_API_KEY or OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_output.jsonl",
        help="Output file path (default: benchmark_output.jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in response (default: 1024, used with --models)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0, used with --models)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent requests (default: 4)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    # Load queries
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: Queries file not found: {queries_path}")
        sys.exit(1)

    queries = load_queries(queries_path)

    if not queries:
        print("Error: No queries loaded")
        sys.exit(1)

    # Load model configs
    if args.model_config:
        # Load from YAML config file
        config_path = Path(args.model_config)
        if not config_path.exists():
            print(f"Error: Model config file not found: {config_path}")
            sys.exit(1)
        model_configs = load_model_configs(config_path)
        print(f"Loaded {len(model_configs)} model configurations from {config_path}")
    else:
        # Create from simple model list
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            print("Error: No models specified")
            sys.exit(1)
        model_configs = create_model_configs_from_list(
            models=models,
            endpoint=args.endpoint,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    if not model_configs:
        print("Error: No model configurations loaded")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(
        queries=queries,
        model_configs=model_configs,
        concurrency=args.concurrency,
        progress=not args.no_progress,
    )

    # Save results
    output_path = Path(args.output)
    save_results(results, output_path)

    # Print summary
    print_summary(results)

    print("Next steps:")
    print("  1. Add categories using VSR classifier:")
    print(
        f"     python add_category_to_training_data.py --input {output_path} --output benchmark_with_category.jsonl"
    )
    print("  2. Train models:")
    print(
        "     python train.py --data-file benchmark_with_category.jsonl --output-dir models/"
    )
    print()


if __name__ == "__main__":
    main()
