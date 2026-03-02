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
    """Model onboarding evaluation class"""

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
            extra_params=config.get("extra_params", {}),
        )

        # Set random seeds
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)

        return self.model_config

    def run_performance_test(self, test_name: str, **kwargs) -> TestResult:
        """
        Run model performance test

        Args:
            test_name: Test name ("arc_challenge" or "mmlu_pro")
            **kwargs: Test parameters
                - samples: number of samples (arc_challenge)
                - samples_per_category: samples per category (mmlu_pro)
                - categories: MMLU-Pro categories

        Returns:
            TestResult: Test result
        """
        if not self.model_config:
            raise ValueError("Please call parse() first to load model config")

        if test_name == "arc_challenge":
            return self._run_arc_challenge(**kwargs)
        elif test_name == "mmlu_pro":
            return self._run_mmlu_pro(**kwargs)
        else:
            raise ValueError(f"Unsupported test type: {test_name}")

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
        overall_accuracy = (
            valid_results["is_correct"].mean() if not valid_results.empty else 0.0
        )

        # Build test result
        test_result = TestResult(
            model_name=self.model_config.model_name,
            test_name="arc_challenge",
            score=overall_accuracy,
            metrics={
                "overall": overall_accuracy,
                "total_questions": len(results_df),
                "successful_queries": len(valid_results),
                "failed_queries": len(results_df) - len(valid_results),
            },
            details={"split": "train", "samples_evaluated": len(df)},
        )

        self.test_results.append(test_result)
        return test_result

    def _run_mmlu_pro(
        self, samples_per_category: int = 5, categories: Optional[List[str]] = None
    ) -> TestResult:
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
                sampled_df = category_df.sample(
                    samples_per_category, random_state=self.model_config.seed
                )
                sampled_dfs.append(sampled_df)
            else:
                sampled_dfs.append(category_df)
        df = pd.concat(sampled_dfs)

        # Evaluate
        results_df = self._evaluate_mmlu(df)
        self.mmlu_results = results_df

        # Analyze
        valid_results = results_df[results_df["success"]]
        overall_accuracy = (
            valid_results["is_correct"].mean() if not valid_results.empty else 0.0
        )

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
                "overall_accuracy": overall_accuracy,
            },
        )

        self.test_results.append(test_result)
        return test_result

    def _evaluate_arc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate ARC"""
        client = OpenAI(
            base_url=self.model_config.endpoint,
            api_key=self.model_config.api_key or "dummy",
        )

        results = []
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Evaluating ARC Challenge"
        ):
            result = self._process_arc_question(client, row.to_dict())
            results.append(result)

        return pd.DataFrame(results)

    def _evaluate_mmlu(self, df: pd.DataFrame) -> pd.DataFrame:
        """Evaluate MMLU"""
        client = OpenAI(
            base_url=self.model_config.endpoint,
            api_key=self.model_config.api_key or "dummy",
        )

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating MMLU-Pro"):
            result = self._process_mmlu_question(client, row.to_dict())
            results.append(result)

        return pd.DataFrame(results)

    def _process_arc_question(
        self, client: OpenAI, question_data: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            "success": True,
        }

    def _process_mmlu_question(
        self, client: OpenAI, question_data: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            9: "J",
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
            "success": True,
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
            output_path = (
                f"{self.model_config.model_name.replace('/', '_')}_{approach}.json"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        return output_path

    def _generate_report(self) -> str:
        """Generate JSON report"""
        arc_result = next(
            (r for r in self.test_results if r.test_name == "arc_challenge"), None
        )
        mmlu_result = next(
            (r for r in self.test_results if r.test_name == "mmlu_pro"), None
        )

        report_data = {
            "model_name": self.model_config.model_name,
            "approach": "Chain-of-Thought" if self.model_config.use_cot else "Direct",
        }

        if arc_result:
            report_data["global_metrics"] = {
                "reasoning": {
                    "benchmark": "ARC-Challenge",
                    "results": {"overall": round(arc_result.score, 3)},
                }
            }

        if mmlu_result:
            report_data["domain_scores"] = {
                "benchmark": "MMLU-Pro",
                "results": {k: round(v, 2) for k, v in mmlu_result.metrics.items()},
            }

        metadata = {
            "test_time": datetime.now().isoformat() + "Z",
            "seed": self.model_config.seed,
        }
        if arc_result and arc_result.details:
            metadata.update(arc_result.details)
        elif mmlu_result and mmlu_result.details:
            metadata.update(mmlu_result.details)

        report_data["metadata"] = metadata
        return json.dumps(report_data, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    onboard = OnboardEvaluate(config_path="config.json")

    config = {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "endpoint": "http://localhost:8000/v1",
        "api_key": "",
        "max_tokens": 128,
        "temperature": 0.0,
        "use_cot": False,
        "seed": 42,
    }
    model_config = onboard.parse(config)
    print(f"Model config parsed: {model_config.model_name}")

    print("\nRunning ARC Challenge test...")
    arc_result = onboard.run_performance_test("arc_challenge", samples=20)
    print(f"ARC Challenge accuracy: {arc_result.score:.4f}")

    print("\nRunning MMLU-Pro test...")
    mmlu_result = onboard.run_performance_test("mmlu_pro", samples_per_category=5)
    print(f"MMLU-Pro accuracy: {mmlu_result.score:.4f}")

    print("\nGenerating report...")
    report = onboard.generate_report()
    print(f"Report saved to: {report}")
