"""
OnboardEvaluate - Model onboarding evaluation framework
Provides model performance testing, parsing, and report generation.
"""

# TODO: Add the structure for performance metrics in TestResult:
# "performance_metrics": {
#     "total_consumed_tokens": 12345,
#     "avg_completion_tokens_per_query": 123,
#     "avg_tps": 12,
#     "avg_latency_ms": 123
# }

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


# Constants
ANSWER_PATTERN_ARC = re.compile(r"(?:answer(?:\sis)?:?\s*)(A|B|C|D)", re.IGNORECASE)
ANSWER_PATTERN_MMLU = re.compile(
    r"(?:answer(?:\sis)?:?\s*)(A|B|C|D|E|F|G|H|I|J)", re.IGNORECASE
)

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    endpoint: str
    api_key: str = ""
    max_tokens: int = 512
    temperature: float = 0.0
    use_cot: bool = False
    seed: int = 42
    extra_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestResult:
    """Test result"""
    model_name: str
    test_name: str
    score: float
    metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = None

class OnboardEvaluate:
    """Model onboarding evaluation class (system-level eval via /api/v1/eval)"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize OnboardEvaluate
        
        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.model_config: Optional[ModelConfig] = None
        self.test_results: List[TestResult] = []
        self.arc_results: Optional[pd.DataFrame] = None
        self.mmlu_results: Optional[pd.DataFrame] = None
        self.system_eval_responses: List[Dict[str, Any]] = []
        self.system_eval_summary: List[Dict[str, Any]] = []
    
    def parse(self, config: Dict[str, Any]) -> ModelConfig:
        """
        Parse model configuration
        
        Args:
            config: Config dict including model name, endpoint, etc.
            
        Returns:
            ModelConfig: Parsed model config
        """
        self.model_config = ModelConfig(
            model_name=config.get("model_name", ""),
            endpoint=config.get("endpoint", ""),
            api_key=config.get("api_key", ""),
            max_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 0.0),
            use_cot=config.get("use_cot", False),
            seed=config.get("seed", 42),
            extra_params=config.get("extra_params", {})
        )
        
        # Set random seeds
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        
        return self.model_config
    
    def run_performance_test(self, test_name: str, **kwargs) -> TestResult:
        """
        Run model performance test
        
        Args:
            test_name: Test name ("arc_challenge", "mmlu_pro", or "system_eval")
            **kwargs: Test parameters
                - samples: number of samples (arc_challenge)
                - samples_per_category: samples per category (mmlu_pro)
                - categories: MMLU-Pro categories
                - texts: list of texts to evaluate (system_eval)
                - options: intent options dict (system_eval)
                - input_path: optional text/jsonl file with one prompt per line (system_eval)
                - datasets: list of dataset IDs for system eval (system_eval)
                - max_samples: max samples per dataset (system_eval)

        Returns:
            TestResult: Test result
        """
        if not self.model_config:
            raise ValueError("Please call parse() first to load model config")
        
        if test_name == "arc_challenge":
            return self._run_arc_challenge(**kwargs)
        elif test_name == "mmlu_pro":
            return self._run_mmlu_pro(**kwargs)
        elif test_name == "system_eval":
            return self._run_system_eval(**kwargs)
        else:
            raise ValueError(f"Unsupported test type: {test_name}")

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
    ) -> (str, Any, Dict[str, Any]):
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
    ) -> (List[Dict[str, Any]], Dict[str, Any]):
        total = 0
        correct = 0
        incorrect = 0
        skipped = 0
        latency_ms: List[float] = []

        results: List[Dict[str, Any]] = []
        for row in tqdm(dataset, desc=f"Evaluating {dataset_id}"):
            text = self._extract_text(row, config)
            expected = self._extract_label(row, config, dataset, dimension)

            response = self._post_eval_request({"text": text})
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
        }
        return results, metrics

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

    def _run_arc_challenge(self, samples: Optional[int] = 20) -> TestResult:
        """Run ARC Challenge test"""
        print(f"Starting ARC Challenge test, samples: {samples}")
        
        # Load dataset
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        df = pd.DataFrame(dataset)
        
        if samples and len(df) > samples:
            df = df.sample(samples, random_state=self.model_config.seed)
        
        # Evaluate
        results_df = self._evaluate_arc(df)
        self.arc_results = results_df
        
        # Analyze
        valid_results = results_df[results_df["success"]]
        overall_accuracy = valid_results["is_correct"].mean() if not valid_results.empty else 0.0
        
        # Build test result
        test_result = TestResult(
            model_name=self.model_config.model_name,
            test_name="arc_challenge",
            score=overall_accuracy,
            metrics={
                "overall": overall_accuracy,
                "total_questions": len(results_df),
                "successful_queries": len(valid_results),
                "failed_queries": len(results_df) - len(valid_results)
            },
            details={
                "split": "train",
                "samples_evaluated": len(df)
            }
        )
        
        self.test_results.append(test_result)
        return test_result
    
    def _run_mmlu_pro(self, samples_per_category: int = 5, 
                      categories: Optional[List[str]] = None) -> TestResult:
        """Run MMLU-Pro test"""
        print(f"Starting MMLU-Pro test, samples per category: {samples_per_category}")
        
        # Load dataset
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        df = pd.DataFrame(dataset)
        
        if categories:
            df = df[df["category"].isin(categories)]
        
        # Sample per category
        sampled_dfs = []
        for category in df["category"].unique():
            category_df = df[df["category"] == category]
            if len(category_df) > samples_per_category:
                sampled_df = category_df.sample(samples_per_category, random_state=self.model_config.seed)
                sampled_dfs.append(sampled_df)
            else:
                sampled_dfs.append(category_df)
        df = pd.concat(sampled_dfs)
        
        # Evaluate
        results_df = self._evaluate_mmlu(df)
        self.mmlu_results = results_df
        
        # Analyze
        valid_results = results_df[results_df["success"]]
        overall_accuracy = valid_results["is_correct"].mean() if not valid_results.empty else 0.0
        
        # Per-category accuracy
        category_accuracy = {}
        for category in valid_results["category"].unique():
            category_df = valid_results[valid_results["category"] == category]
            category_accuracy[category] = category_df["is_correct"].mean()
        
        # Build test result
        test_result = TestResult(
            model_name=self.model_config.model_name,
            test_name="mmlu_pro",
            score=overall_accuracy,
            metrics=category_accuracy,
            details={
                "split": "test",
                "samples_evaluated": len(df),
                "overall_accuracy": overall_accuracy
            }
        )
        
        self.test_results.append(test_result)
        return test_result
    
    def _evaluate_arc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate ARC"""
        client = OpenAI(
            base_url=self.model_config.endpoint,
            api_key=self.model_config.api_key or "dummy"
        )
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating ARC Challenge"):
            result = self._process_arc_question(client, row.to_dict())
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _evaluate_mmlu(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate MMLU"""
        client = OpenAI(
            base_url=self.model_config.endpoint,
            api_key=self.model_config.api_key or "dummy"
        )
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating MMLU-Pro"):
            result = self._process_mmlu_question(client, row.to_dict())
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _process_arc_question(self, client: OpenAI, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single ARC question"""
        question = question_data["question"]
        choices = question_data["choices"]
        correct_answer = question_data["answerKey"]
        
        # Format prompt
        formatted_options = ""
        for label, text in zip(choices["label"], choices["text"]):
            formatted_options += f"{label}) {text}\n"
        
        if self.model_config.use_cot:
            prompt = f"Question: {question}\n\nOptions:\n{formatted_options}\n\nPlease solve this step-by-step, then provide your final answer in the format 'Answer: [letter]'."
        else:
            prompt = f"Question: {question}\n\nOptions:\n{formatted_options}\n\nPlease choose the correct answer from the options above. Provide your answer in the format 'Answer: [letter]'."
        
        # Call model
        response = client.chat.completions.create(
            model=self.model_config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.model_config.max_tokens,
            temperature=self.model_config.temperature,
        )
        response_text = response.choices[0].message.content
        
        # Extract answer
        predicted_answer = self._extract_answer_arc(response_text)
        is_correct = predicted_answer == correct_answer
        
        return {
            "id": question_data["id"],
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "success": True
        }
    
    def _process_mmlu_question(self, client: OpenAI, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single MMLU question"""
        question = question_data["question"]
        options = question_data["options"]
        correct_answer = question_data["answer"]
        category = question_data["category"]
        
        # Format prompt
        letter_mapping = {
            0: "A", 
            1: "B", 
            2: "C", 
            3: "D", 
            4: "E", 
            5: "F", 
            6: "G", 
            7: "H", 
            8: "I", 
            9: "J"
        }
        formatted_options = ""
        for i, option in enumerate(options):
            if option.lower() != "n/a":
                formatted_options += f"{letter_mapping[i]}) {option}\n"
        
        if self.model_config.use_cot:
            prompt = f"Question: {question}\n\nOptions:\n{formatted_options}\n\nPlease solve this step-by-step, then provide your final answer in the format 'Answer: [letter]'."
        else:
            prompt = f"Question: {question}\n\nOptions:\n{formatted_options}\n\nPlease choose the correct answer from the options above. Provide your answer in the format 'Answer: [letter]'."
        
        # Call model
        response = client.chat.completions.create(
            model=self.model_config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.model_config.max_tokens,
            temperature=self.model_config.temperature,
        )
        response_text = response.choices[0].message.content
        
        # Extract answer
        predicted_answer = self._extract_answer_mmlu(response_text)
        is_correct = predicted_answer == correct_answer
        
        return {
            "question_id": question_data["question_id"],
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "category": category,
            "success": True
        }
    
    def _extract_answer_arc(self, response: str) -> Optional[str]:
        """Extract ARC answer from response"""
        match = ANSWER_PATTERN_ARC.search(response)
        if match:
            return match.group(1).upper()
        
        for char in reversed(response):
            if char.upper() in "ABCD":
                return char.upper()
        
        return None
    
    def _extract_answer_mmlu(self, response: str) -> Optional[str]:
        """Extract MMLU answer from response"""
        match = ANSWER_PATTERN_MMLU.search(response)
        if match:
            return match.group(1).upper()
        
        for char in reversed(response):
            if char.upper() in "ABCDEFGHIJ":
                return char.upper()
        
        return None
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate test report
        
        Args:
            output_path: output path; if None, auto-generate filename
            
        Returns:
            str: report file path
        """
        if not self.test_results:
            raise ValueError("No test results available")
        
        report = self._generate_report()
        
        if output_path is None:
            approach = "CoT" if self.model_config.use_cot else "Direct"
            output_path = f"{self.model_config.model_name.replace('/', '_')}_{approach}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return output_path
    
    def _generate_report(self) -> str:
        """Generate JSON report"""
        if self.system_eval_responses:
            overall_score = next(
                (r.score for r in self.test_results if r.test_name == "system_eval"),
                0.0,
            )
            report_data = {
                "endpoint": self._normalize_eval_endpoint(),
                "generated_at": datetime.now().isoformat() + "Z",
                "overall_score": overall_score,
                "summary": self.system_eval_summary,
                "results": self.system_eval_responses,
            }
            return json.dumps(report_data, indent=4, ensure_ascii=False)

        arc_result = next(
            (r for r in self.test_results if r.test_name == "arc_challenge"), None
        )
        mmlu_result = next(
            (r for r in self.test_results if r.test_name == "mmlu_pro"), None
        )

        report_data = {
            "model_name": self.model_config.model_name,
            "approach": "Chain-of-Thought" if self.model_config.use_cot else "Direct"
        }
        
        if arc_result:
            report_data["global_metrics"] = {
                "reasoning": {
                    "benchmark": "ARC-Challenge",
                    "results": {"overall": round(arc_result.score, 3)}
                }
            }
        
        if mmlu_result:
            report_data["domain_scores"] = {
                "benchmark": "MMLU-Pro",
                "results": {k: round(v, 2) for k, v in mmlu_result.metrics.items()}
            }
        
        metadata = {"test_time": datetime.now().isoformat() + "Z", "seed": self.model_config.seed}
        if arc_result and arc_result.details:
            metadata.update(arc_result.details)
        elif mmlu_result and mmlu_result.details:
            metadata.update(mmlu_result.details)
        
        report_data["metadata"] = metadata
        return json.dumps(report_data, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    onboard = OnboardEvaluate(config_path="config.json")
    
    config = {
        "model_name": "router-system-eval",
        "endpoint": "http://localhost:8080",
        "api_key": "",
        "max_tokens": 128,
        "temperature": 0.0,
        "use_cot": False,
        "seed": 42
    }
    model_config = onboard.parse(config)
    print(f"Model config parsed: {model_config.model_name}")

    print("\nRunning system-level evaluation...")
    system_result = onboard.run_performance_test(
        "system_eval",
        datasets=["mmlu-pro-en", "mmlu-prox-zh", "fact-check-en", "feedback-en"],
        max_samples=50,
    )
    print(
        f"System eval completed, samples: {system_result.metrics.get('total_samples', 0)}"
    )

    print("\nGenerating report...")
    report = onboard.generate_report()
    print(f"Report saved to: {report}")