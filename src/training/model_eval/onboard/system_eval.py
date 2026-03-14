import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from .constants import SYSTEM_MATCHED_SIGNAL_KEYS, SYSTEM_SIGNAL_KEYS
from .types import TestResult


class SystemEvalMixin:
    def _run_system_eval(
        self,
        texts: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        input_path: Optional[str] = None,
        datasets: Optional[List[str]] = None,
        max_samples: int = 50,
    ) -> TestResult:
        """Run system-level evaluation via /api/v1/eval"""
        if texts is None:
            texts = []

        if datasets:
            return self._run_system_eval_datasets(datasets, max_samples=max_samples)

        if input_path:
            texts.extend(self._load_texts_from_file(input_path))

        if not texts:
            raise ValueError("system_eval requires a non-empty 'texts' list")

        if options is None:
            options = {"return_probabilities": False, "include_explanation": True}

        responses: List[Dict[str, Any]] = []
        for text in tqdm(texts, total=len(texts), desc="Evaluating System (API)"):
            payload: Dict[str, Any] = {"text": text}
            payload["options"] = options
            response = self._post_eval_request(payload)
            responses.append({"input": text, "response": response})

        self.system_eval_responses = responses

        test_result = TestResult(
            model_name=self.model_config.model_name,
            test_name="system_eval",
            score=0.0,
            metrics={"total_samples": len(texts)},
            details={"endpoint": self._normalize_eval_endpoint()},
        )

        self.test_results.append(test_result)
        return test_result

    def _normalize_eval_endpoint(self) -> str:
        endpoint = self.model_config.endpoint.rstrip("/")
        if endpoint.endswith("/api/v1/eval"):
            return endpoint
        return endpoint + "/api/v1/eval"

    def _post_eval_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST request to /api/v1/eval and return raw JSON response"""
        if not self.model_config:
            raise ValueError("Please call parse() first to load model config")

        endpoint = self._normalize_eval_endpoint()
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8") if exc.fp else ""
            raise RuntimeError(
                f"Eval request failed ({exc.code}): {error_body}"
            ) from exc

        return json.loads(body) if body else {}

    def _run_system_eval_datasets(
        self, datasets: List[str], max_samples: int = 50
    ) -> TestResult:
        """Run system-level evaluation over curated datasets."""
        if not datasets:
            raise ValueError("datasets must be a non-empty list")

        summary: List[Dict[str, Any]] = []
        all_results: List[Dict[str, Any]] = []

        for dataset_id in datasets:
            dimension, dataset, config = self._load_dataset_rows(
                dataset_id, max_samples
            )
            dataset_results, metrics = self._evaluate_dataset_rows(
                dataset_id, dimension, dataset, config
            )
            summary.append(metrics)
            all_results.extend(dataset_results)

        self.system_eval_responses = all_results

        overall_score = self._compute_overall_score(summary)
        total_samples = sum(item.get("total_samples", 0) for item in summary)
        test_result = TestResult(
            model_name=self.model_config.model_name,
            test_name="system_eval",
            score=overall_score,
            metrics={"overall_score": overall_score, "total_samples": total_samples},
            details={"endpoint": self._normalize_eval_endpoint()},
        )

        self.test_results.append(test_result)
        self.system_eval_summary = summary
        return test_result

    def _compute_overall_score(self, summary: List[Dict[str, Any]]) -> float:
        accuracies = [item.get("accuracy") for item in summary if item.get("accuracy")]
        if not accuracies:
            return 0.0
        return float(sum(accuracies) / len(accuracies))

    def _load_dataset_rows(
        self, dataset_id: str, max_samples: int
    ) -> Tuple[str, Any, Dict[str, Any]]:
        dataset_registry = {
            "mmlu-pro-en": {
                "repo": "TIGER-Lab/MMLU-Pro",
                "split": "test",
                "text_field": "question",
                "label_field": "category",
                "dimension": "domain",
            },
            "mmlu-prox-zh": {
                "repo": "li-lab/MMLU-ProX",
                "config": "zh",
                "split": "test",
                "text_field": "question",
                "label_field": "category",
                "dimension": "domain",
            },
            "fact-check-en": {
                "repo": "llm-semantic-router/fact-check-classification-dataset",
                "config": "en",
                "split": "test",
                "text_field": "text",
                "label_field": "label",
                "dimension": "fact_check",
            },
            "feedback-en": {
                "repo": "llm-semantic-router/feedback-detector-dataset",
                "config": "en",
                "split": "test",
                "text_field": "text",
                "label_field": "label_name",
                "dimension": "user_feedback",
            },
        }

        if dataset_id not in dataset_registry:
            raise ValueError(f"Unsupported dataset: {dataset_id}")

        config = dataset_registry[dataset_id]
        repo = config["repo"]
        split = config["split"]
        config_name = config.get("config")

        dataset = self._load_dataset_with_fallback(repo, config_name, split)

        if max_samples > 0 and len(dataset) > max_samples:
            dataset = dataset.shuffle(seed=self.model_config.seed).select(
                range(max_samples)
            )

        return config["dimension"], dataset, config

    def _load_dataset_with_fallback(
        self, repo: str, config_name: Optional[str], split: str
    ) -> Any:
        """Load dataset with config and split fallbacks."""
        split_candidates = [split, "validation", "train"]

        def load_with_split(cfg: Optional[str], split_name: str) -> Any:
            if cfg:
                return load_dataset(repo, cfg, split=split_name)
            return load_dataset(repo, split=split_name)

        last_error: Optional[Exception] = None

        for split_name in split_candidates:
            try:
                return load_with_split(config_name, split_name)
            except Exception as exc:
                last_error = exc
                try:
                    if config_name:
                        return load_with_split(None, split_name)
                except Exception as fallback_exc:
                    last_error = fallback_exc

        if last_error:
            raise last_error
        raise ValueError("Failed to load dataset")

    def _evaluate_dataset_rows(
        self, dataset_id: str, dimension: str, dataset: Any, config: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        total = 0
        correct = 0
        incorrect = 0
        skipped = 0
        latency_ms: List[float] = []
        signal_stats = self._init_signal_stats()

        results: List[Dict[str, Any]] = []
        for row in tqdm(dataset, desc=f"Evaluating {dataset_id}"):
            text = self._extract_text(row, config)
            expected = self._extract_label(row, config, dataset, dimension)

            response = self._post_eval_request({"text": text})
            self._collect_signal_stats(signal_stats, response)
            predicted = self._extract_prediction(response, dimension)

            is_correct = None
            if expected is None or predicted is None:
                skipped += 1
            else:
                is_correct = expected == predicted
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1

            total += 1
            latency = self._extract_latency(response, dimension)
            if latency is not None:
                latency_ms.append(latency)

            results.append(
                {
                    "dataset": dataset_id,
                    "dimension": dimension,
                    "input": text,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                    "response": response,
                }
            )

        accuracy = float(correct / total) if total else 0.0
        avg_latency = float(sum(latency_ms) / len(latency_ms)) if latency_ms else 0.0
        metrics = {
            "dataset": dataset_id,
            "dimension": dimension,
            "total_samples": total,
            "correct": correct,
            "incorrect": incorrect,
            "skipped": skipped,
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "signal_coverage": self._finalize_signal_stats(signal_stats, total),
        }
        return results, metrics

    def _init_signal_stats(self) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for signal in SYSTEM_SIGNAL_KEYS:
            stats[signal] = {
                "matched_samples": 0.0,
                "metric_available_samples": 0.0,
                "confidence_samples": 0.0,
                "confidence_nonzero_samples": 0.0,
                "latency_sum_ms": 0.0,
                "confidence_sum": 0.0,
            }
        return stats

    def _collect_signal_stats(
        self, signal_stats: Dict[str, Dict[str, float]], response: Dict[str, Any]
    ) -> None:
        decision = response.get("decision_result") or {}
        matched = decision.get("matched_signals") or {}
        metrics = response.get("metrics") or {}

        for signal in SYSTEM_SIGNAL_KEYS:
            matched_key = SYSTEM_MATCHED_SIGNAL_KEYS.get(signal, signal)
            matched_values = matched.get(matched_key) or []
            if matched_values:
                signal_stats[signal]["matched_samples"] += 1.0

            metric_entry = metrics.get(signal) or {}
            if not metric_entry:
                continue

            latency = metric_entry.get("execution_time_ms")
            if latency is not None:
                signal_stats[signal]["metric_available_samples"] += 1.0
                signal_stats[signal]["latency_sum_ms"] += float(latency)

            confidence = metric_entry.get("confidence")
            if confidence is not None:
                signal_stats[signal]["confidence_samples"] += 1.0
                signal_stats[signal]["confidence_sum"] += float(confidence)
                if float(confidence) > 0.0:
                    signal_stats[signal]["confidence_nonzero_samples"] += 1.0

    def _finalize_signal_stats(
        self, signal_stats: Dict[str, Dict[str, float]], total_samples: int
    ) -> Dict[str, Dict[str, float]]:
        total = float(total_samples) if total_samples > 0 else 1.0
        result: Dict[str, Dict[str, float]] = {}
        for signal, stats in signal_stats.items():
            metric_count = stats["metric_available_samples"]
            confidence_count = stats["confidence_samples"]

            avg_latency = 0.0
            if metric_count > 0:
                avg_latency = stats["latency_sum_ms"] / metric_count

            avg_confidence = 0.0
            if confidence_count > 0:
                avg_confidence = stats["confidence_sum"] / confidence_count

            result[signal] = {
                "matched_samples": int(stats["matched_samples"]),
                "match_rate": stats["matched_samples"] / total,
                "metric_available_samples": int(metric_count),
                "metric_coverage_rate": metric_count / total,
                "avg_latency_ms": avg_latency,
                "confidence_nonzero_samples": int(stats["confidence_nonzero_samples"]),
                "confidence_nonzero_rate": stats["confidence_nonzero_samples"] / total,
                "avg_confidence": avg_confidence,
            }
        return result

    def _extract_text(self, row: Dict[str, Any], config: Dict[str, Any]) -> str:
        for field in [
            config.get("text_field"),
            "text",
            "prompt",
            "question",
            "input",
        ]:
            if field and field in row and row[field]:
                return str(row[field])
        raise ValueError("No text field found in dataset row")

    def _extract_label(
        self,
        row: Dict[str, Any],
        config: Dict[str, Any],
        dataset: Any,
        dimension: str,
    ) -> Optional[str]:
        label_field = config.get("label_field")
        label_value = None
        if label_field and label_field in row:
            label_value = row[label_field]
        else:
            for field in ["label", "category", "domain", "class"]:
                if field in row:
                    label_field = field
                    label_value = row[field]
                    break

        if label_value is None:
            return None

        label_name = self._label_value_to_name(dataset, label_field, label_value)

        if dimension == "fact_check":
            return self._normalize_fact_check_label(label_name)
        if dimension == "user_feedback":
            return self._normalize_feedback_label(label_name)

        return self._normalize_label(label_name)

    def _label_value_to_name(
        self, dataset: Any, field: Optional[str], value: Any
    ) -> str:
        if field and hasattr(dataset, "features") and field in dataset.features:
            feature = dataset.features[field]
            if hasattr(feature, "names") and isinstance(value, int):
                names = feature.names
                if 0 <= value < len(names):
                    return str(names[value])

        if isinstance(value, (int, float)):
            return str(int(value))
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def _normalize_label(self, label: str) -> str:
        return str(label).strip().lower().replace(" ", "_")

    def _normalize_fact_check_label(self, label: str) -> str:
        normalized = self._normalize_label(label)
        if normalized in {"1", "true", "needs_fact_check", "fact_check_needed"}:
            return "needs_fact_check"
        if normalized in {"0", "false", "no_fact_check_needed", "no_fact_check"}:
            return "no_fact_check_needed"
        if "need" in normalized or "fact" in normalized:
            return "needs_fact_check"
        return "no_fact_check_needed"

    def _normalize_feedback_label(self, label: str) -> str:
        normalized = self._normalize_label(label)
        mapping = {
            "sat": "satisfied",
            "need_clarification": "need_clarification",
            "clarification": "need_clarification",
            "satisfied": "satisfied",
            "want_different": "want_different",
            "different": "want_different",
            "wrong_answer": "wrong_answer",
            "wrong": "wrong_answer",
        }
        return mapping.get(normalized, normalized)

    def _extract_prediction(
        self, response: Dict[str, Any], dimension: str
    ) -> Optional[str]:
        decision = response.get("decision_result") or {}
        matched = decision.get("matched_signals") or {}
        key = {
            "domain": "domains",
            "fact_check": "fact_check",
            "user_feedback": "user_feedback",
        }.get(dimension)

        if not key:
            return None

        values = matched.get(key) or []
        if not values:
            return None
        return self._normalize_label(values[0])

    def _extract_latency(
        self, response: Dict[str, Any], dimension: str
    ) -> Optional[float]:
        metrics = response.get("metrics") or {}
        metric_key = {
            "domain": "domain",
            "fact_check": "fact_check",
            "user_feedback": "user_feedback",
        }.get(dimension)
        if not metric_key:
            return None
        entry = metrics.get(metric_key) or {}
        value = entry.get("execution_time_ms")
        if value is None:
            return None
        return float(value)

    def _load_texts_from_file(self, input_path: str) -> List[str]:
        """Load texts from a txt/jsonl file (one prompt per line)."""
        texts: List[str] = []
        with open(input_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("{"):
                    try:
                        record = json.loads(stripped)
                        text = record.get("text") or record.get("prompt")
                        if text:
                            texts.append(str(text))
                    except json.JSONDecodeError:
                        texts.append(stripped)
                else:
                    texts.append(stripped)
        return texts
