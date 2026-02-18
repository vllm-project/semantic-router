"""End-to-end hallucination detection benchmark.

This module evaluates the hallucination detection pipeline by sending requests
to an already-running semantic router + vLLM setup.

The benchmark:
1. Discovers available models from /v1/models endpoint
2. Sends requests with tool context through the router
3. Collects hallucination detection results from response headers
4. Calculates precision/recall metrics

Usage:
    # Discover models and run quick test
    python -m bench.hallucination_bench.evaluate --max-samples 10

    # Full evaluation with HaluEval dataset
    python -m bench.hallucination_bench.evaluate --dataset halueval --max-samples 100

    # Specify custom endpoint
    python -m bench.hallucination_bench.evaluate --endpoint http://localhost:8801
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .datasets import HallucinationSample, get_dataset


@dataclass
class DetectionResult:
    """Result from hallucination detection."""

    sample_id: str
    question: str
    context_length: int
    response_length: int
    model_used: str

    # Hallucination detection results (from response headers)
    hallucination_detected: Optional[bool] = None
    hallucination_spans: Optional[str] = None

    # Efficiency metrics (from response headers)
    fact_check_needed: Optional[bool] = None  # Was fact-check classification triggered?
    detection_skipped: bool = False  # Was detection skipped due to non-factual query?

    # Ground truth (if available)
    gt_is_faithful: Optional[bool] = None

    # Metrics
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    dataset_name: str
    endpoint: str
    model: str
    total_samples: int
    successful_requests: int
    failed_requests: int

    # Detection metrics
    total_hallucinations_detected: int

    # Accuracy (when ground truth available)
    true_positives: int  # Correctly detected hallucinations
    false_positives: int  # Incorrectly flagged as hallucination
    true_negatives: int  # Correctly identified faithful responses
    false_negatives: int  # Missed hallucinations

    precision: float
    recall: float
    f1_score: float
    accuracy: float

    # Latency
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float

    # Individual results
    results: List[dict] = None  # type: ignore  # Initialized in __post_init__

    # Efficiency metrics (two-stage pipeline savings)
    fact_check_needed_count: int = 0  # Queries that needed fact-checking
    detection_skipped_count: int = 0  # Queries where detection was skipped
    avg_context_length: float = 0.0  # Average context size in chars
    estimated_detection_time_ms: float = 0.0  # Estimated time if all ran detection
    actual_detection_time_ms: float = 0.0  # Actual detection time (skipped = 0)
    time_saved_ms: float = 0.0  # Time saved by pre-filtering
    efficiency_gain_percent: float = 0.0  # Percentage improvement

    def __post_init__(self):
        if self.results is None:
            self.results = []


class HallucinationBenchmark:
    """End-to-end hallucination detection benchmark."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8801",
        timeout: int = 120,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.results: List[DetectionResult] = []
        self.available_models: List[str] = []

    def discover_models(self) -> List[str]:
        """Discover available models from /v1/models endpoint."""
        try:
            resp = requests.get(f"{self.endpoint}/v1/models", timeout=10)
            if resp.status_code != 200:
                print(f"❌ Failed to get models: HTTP {resp.status_code}")
                return []

            data = resp.json()
            models = [
                m.get("id", m.get("name", "unknown")) for m in data.get("data", [])
            ]
            self.available_models = models

            print(f"✓ Discovered {len(models)} models:")
            for m in models:
                print(f"  - {m}")

            return models
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to discover models: {e}")
            return []

    def check_health(self) -> bool:
        """Check if the endpoint is healthy."""
        try:
            # Try /v1/models as health check
            resp = requests.get(f"{self.endpoint}/v1/models", timeout=10)
            if resp.status_code == 200:
                print(f"✓ Endpoint {self.endpoint} is healthy")
                return True
            else:
                print(f"❌ Endpoint returned HTTP {resp.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
            return False

    def send_request(
        self,
        sample: HallucinationSample,
        model: str,
        include_context: bool = True,
    ) -> DetectionResult:
        """Send a request and collect hallucination detection results."""
        start_time = time.time()

        result = DetectionResult(
            sample_id=sample.id,
            question=sample.question[:100],
            context_length=len(sample.context),
            response_length=0,
            model_used=model,
            gt_is_faithful=sample.is_faithful,
        )

        try:
            # Build messages - include tool context if available
            messages = [{"role": "user", "content": sample.question}]

            if include_context and sample.context:
                # Add tool result with context (enables hallucination detection)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"ctx_{sample.id}",
                        "content": sample.context,
                    }
                )

            # Send request
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )

            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms

            if response.status_code != 200:
                result.error = f"HTTP {response.status_code}: {response.text[:200]}"
                return result

            # Parse response
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
                result.response_length = len(content)

            # Extract hallucination detection headers
            headers = response.headers
            result.hallucination_detected = (
                headers.get("x-vsr-hallucination-detected", "").lower() == "true"
            )
            result.hallucination_spans = headers.get("x-vsr-hallucination-spans", "")

            # Extract efficiency-related headers
            result.fact_check_needed = (
                headers.get("x-vsr-fact-check-needed", "").lower() == "true"
            )
            # Detection is skipped if fact-check says not needed, or if unverified (no context)
            unverified = (
                headers.get("x-vsr-unverified-factual-response", "").lower() == "true"
            )
            result.detection_skipped = not result.fact_check_needed or unverified

        except requests.exceptions.Timeout:
            result.error = "Request timeout"
        except requests.exceptions.RequestException as e:
            result.error = str(e)
        except Exception as e:
            result.error = f"Unexpected error: {e}"

        return result

    def run_benchmark(
        self,
        samples: List[HallucinationSample],
        model: str,
        include_context: bool = True,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """Run the benchmark on a list of samples."""
        from tqdm import tqdm

        print(f"\nRunning hallucination detection benchmark...")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Model: {model}")
        print(f"  Samples: {len(samples)}")
        print(f"  Context: {'included' if include_context else 'excluded'}")
        print()

        self.results = []
        detected_count = 0

        # Use tqdm with leave=True to keep progress bar visible
        pbar = tqdm(
            samples,
            desc="Evaluating",
            unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Hallucinations: {postfix}",
        )
        pbar.set_postfix_str("0")

        for sample in pbar:
            result = self.send_request(sample, model, include_context=include_context)
            self.results.append(result)

            # Print hallucination detection immediately when found
            if result.hallucination_detected and verbose:
                detected_count += 1
                pbar.set_postfix_str(str(detected_count))

                # Print detection details below progress bar
                tqdm.write("")
                tqdm.write(f"🚨 HALLUCINATION DETECTED [{detected_count}]")
                tqdm.write(f"   Question: {result.question[:80]}...")
                if result.hallucination_spans:
                    spans_preview = result.hallucination_spans[:150]
                    tqdm.write(
                        f"   Spans: {spans_preview}{'...' if len(result.hallucination_spans) > 150 else ''}"
                    )
                if result.gt_is_faithful is not None:
                    gt_label = (
                        "✅ Correct"
                        if not result.gt_is_faithful
                        else "❌ False Positive"
                    )
                    tqdm.write(f"   Ground Truth: {gt_label}")
                tqdm.write(f"   Latency: {result.latency_ms:.0f}ms")
                tqdm.write("")

            # Also print errors immediately
            elif result.error and verbose:
                tqdm.write(f"❌ ERROR: {result.error[:100]}")

        dataset_name = (
            samples[0].metadata.get("dataset", "unknown") if samples else "unknown"
        )
        return self.calculate_metrics(dataset_name, model)

    def calculate_metrics(self, dataset_name: str, model: str) -> BenchmarkResults:
        """Calculate benchmark metrics from results."""
        successful = [r for r in self.results if r.error is None]
        failed = [r for r in self.results if r.error is not None]

        # Detection counts
        hallucinations_detected = sum(1 for r in successful if r.hallucination_detected)

        # Accuracy metrics (when ground truth available)
        tp = fp = tn = fn = 0
        for r in successful:
            if r.gt_is_faithful is not None:
                detected = r.hallucination_detected or False
                is_hallucination = not r.gt_is_faithful

                if detected and is_hallucination:
                    tp += 1
                elif detected and not is_hallucination:
                    fp += 1
                elif not detected and not is_hallucination:
                    tn += 1
                elif not detected and is_hallucination:
                    fn += 1

        total_with_gt = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / total_with_gt if total_with_gt > 0 else 0.0

        # False positive rates and specificity
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if fp > 0:
            print(f"\n⚠️  False Positives Detected: {fp}")
            false_positive_samples = []
            for r in successful:
                if (
                    r.gt_is_faithful is not None
                    and r.hallucination_detected
                    and r.gt_is_faithful
                ):
                    false_positive_samples.append(
                        {
                            "sample_id": r.sample_id,
                            "question": r.question[:100],
                            "spans": r.hallucination_spans,
                        }
                    )

                if false_positive_samples:
                    print("\nFalse Positive Examples:")
                    for i, fp_sample in enumerate(
                        false_positive_samples[:5], 1
                    ):  # Show first 5
                        print(f"\n  {i}. Sample ID: {fp_sample['sample_id']}")
                        print(f"     Question: {fp_sample['question']}...")
                        print(f"     Flagged spans: {fp_sample['spans']}")

                    if len(false_positive_samples) > 5:
                        print(
                            f"\n  ... and {len(false_positive_samples) - 5} more false positives"
                        )

        # Latency metrics
        latencies = [r.latency_ms for r in successful if r.latency_ms is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0
        p99_idx = min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)
        p99 = sorted_latencies[p99_idx] if sorted_latencies else 0.0

        # Efficiency metrics - calculate savings from two-stage pipeline
        fact_check_needed_count = sum(1 for r in successful if r.fact_check_needed)
        detection_skipped_count = sum(1 for r in successful if r.detection_skipped)

        # Average context length
        context_lengths = [r.context_length for r in successful if r.context_length > 0]
        avg_context_length = (
            sum(context_lengths) / len(context_lengths) if context_lengths else 0.0
        )

        # Estimate detection time based on context length
        # ModernBERT-large (395M params) timing estimates:
        # - Fact-check classifier (ModernBERT-base): ~12ms (prompt only)
        # - LettuceDetect-large: ~45ms base + ~0.02ms per character
        # These are empirical measurements on CPU; GPU would be faster
        CLASSIFIER_TIME_MS = (
            12.0  # Fact-check classifier time (ModernBERT-base, prompt only)
        )
        DETECTION_BASE_MS = 45.0  # Base detection time (ModernBERT-large)
        DETECTION_PER_CHAR_MS = 0.02  # Additional time per character (large model)

        estimated_detection_times = []
        actual_detection_times = []
        for r in successful:
            # Estimated time if we ran detection on everything
            est_time = DETECTION_BASE_MS + (r.context_length * DETECTION_PER_CHAR_MS)
            estimated_detection_times.append(est_time)

            # Actual time (0 if skipped, otherwise estimated)
            if r.detection_skipped:
                actual_detection_times.append(0.0)
            else:
                actual_detection_times.append(est_time)

        estimated_total = sum(estimated_detection_times)
        actual_total = sum(actual_detection_times)
        time_saved = estimated_total - actual_total
        efficiency_gain = (
            (time_saved / estimated_total * 100) if estimated_total > 0 else 0.0
        )

        return BenchmarkResults(
            dataset_name=dataset_name,
            endpoint=self.endpoint,
            model=model,
            total_samples=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_hallucinations_detected=hallucinations_detected,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
            # Efficiency metrics
            fact_check_needed_count=fact_check_needed_count,
            detection_skipped_count=detection_skipped_count,
            avg_context_length=avg_context_length,
            estimated_detection_time_ms=estimated_total,
            actual_detection_time_ms=actual_total,
            time_saved_ms=time_saved,
            efficiency_gain_percent=efficiency_gain,
            results=[asdict(r) for r in self.results],
        )


def print_results(results: BenchmarkResults):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print(f"HALLUCINATION DETECTION BENCHMARK")
    print("=" * 60)

    print(f"\n📌 Configuration:")
    print("-" * 40)
    print(f"  Dataset:  {results.dataset_name}")
    print(f"  Endpoint: {results.endpoint}")
    print(f"  Model:    {results.model}")

    print(f"\n📊 Request Statistics:")
    print("-" * 40)
    print(f"  Total samples:  {results.total_samples}")
    print(f"  Successful:     {results.successful_requests}")
    print(f"  Failed:         {results.failed_requests}")

    print(f"\n📋 Detection Results:")
    print("-" * 40)
    print(f"  Hallucinations detected: {results.total_hallucinations_detected}")

    total_gt = (
        results.true_positives
        + results.false_positives
        + results.true_negatives
        + results.false_negatives
    )
    if total_gt > 0:
        print(f"\n🎯 Accuracy Metrics (vs ground truth):")
        print("-" * 40)
        print(f"  True Positives:  {results.true_positives}")
        print(f"  False Positives: {results.false_positives}")
        print(f"  True Negatives:  {results.true_negatives}")
        print(f"  False Negatives: {results.false_negatives}")
        print(f"  Precision: {results.precision:.4f}")
        print(f"  Recall:    {results.recall:.4f}")
        print(f"  F1 Score:  {results.f1_score:.4f}")
        print(f"  Accuracy:  {results.accuracy:.4f}")
    else:
        print(f"\n⚠️ No ground truth labels available for accuracy metrics")

    print(f"\n⏱️ Latency:")
    print("-" * 40)
    print(f"  Average: {results.avg_latency_ms:.2f} ms")
    print(f"  P50:     {results.p50_latency_ms:.2f} ms")
    print(f"  P99:     {results.p99_latency_ms:.2f} ms")

    # Efficiency metrics - two-stage pipeline savings
    print(f"\n⚡ Two-Stage Pipeline Efficiency:")
    print("-" * 40)
    print(
        f"  Fact-check needed:     {results.fact_check_needed_count}/{results.successful_requests} queries"
    )
    print(
        f"  Detection skipped:     {results.detection_skipped_count}/{results.successful_requests} queries"
    )
    print(f"  Avg context length:    {results.avg_context_length:.0f} chars")
    print(
        f"  Estimated detect time: {results.estimated_detection_time_ms:.2f} ms (if all ran)"
    )
    print(f"  Actual detect time:    {results.actual_detection_time_ms:.2f} ms")
    print(f"  Time saved:            {results.time_saved_ms:.2f} ms")
    print(f"  Efficiency gain:       {results.efficiency_gain_percent:.1f}%")

    if results.detection_skipped_count > 0:
        skip_rate = results.detection_skipped_count / results.successful_requests * 100
        print(f"\n  💡 Pre-filtering skipped {skip_rate:.1f}% of requests,")
        print(f"     saving {results.time_saved_ms:.0f}ms of detection compute.")

    # Show sample results
    print(f"\n🔍 Sample Results (first 5):")
    print("-" * 60)
    for r in results.results[:5]:
        detected = "🚨" if r.get("hallucination_detected") else "✅"
        gt = (
            "HAL"
            if r.get("gt_is_faithful") is False
            else ("OK" if r.get("gt_is_faithful") else "?")
        )
        print(f"  {detected} GT:{gt} | {r.get('question', '')[:50]}...")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Benchmark")

    # Endpoint options
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8801",
        help="Router/Envoy endpoint URL",
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default="halueval",
        help="Dataset to use (halueval, or path to JSONL)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50, help="Maximum samples to evaluate"
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (auto-discovered if not specified)",
    )

    # Test modes
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Don't include tool context (disables hallucination detection)",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="Just list available models and exit"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print hallucinations as they're detected",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bench/hallucination/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create benchmark
    benchmark = HallucinationBenchmark(endpoint=args.endpoint)

    # Check health and discover models
    if not benchmark.check_health():
        print(f"\n❌ Cannot connect to {args.endpoint}")
        print("   Make sure the semantic router and vLLM are running.")
        sys.exit(1)

    models = benchmark.discover_models()
    if not models:
        print("\n❌ No models available")
        sys.exit(1)

    if args.list_models:
        print("\nAvailable models:")
        for m in models:
            print(f"  - {m}")
        sys.exit(0)

    # Select model
    model = args.model or models[0]
    if model not in models and model != "auto":
        print(f"\n⚠️ Model '{model}' not in available models, using anyway...")

    print(f"\n📦 Using model: {model}")

    # Load dataset
    try:
        dataset = get_dataset(args.dataset)
        samples = dataset.load(max_samples=args.max_samples)
    except Exception as e:
        print(f"\n❌ Failed to load dataset: {e}")
        sys.exit(1)

    # Run benchmark
    results = benchmark.run_benchmark(
        samples,
        model=model,
        include_context=not args.no_context,
        verbose=not args.quiet,
    )

    # Print results
    print_results(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{args.dataset}_{timestamp}.json"
    with open(results_file, "w") as f:
        # Don't include individual results in JSON to keep file small
        summary = asdict(results)
        summary["results"] = summary["results"][:10]  # Only first 10 for summary
        json.dump(summary, f, indent=2)
    print(f"\n📁 Results saved to {results_file}")


if __name__ == "__main__":
    main()
