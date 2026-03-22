from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from .constants import ANSWER_PATTERN_MMLU
from .types import TestResult


class MmluEvalMixin:
    def _run_mmlu_pro(
        self, samples_per_category: int = 5, categories: Optional[List[str]] = None
    ) -> TestResult:
        """Run MMLU-Pro test"""
        print(f"Starting MMLU-Pro test, samples per category: {samples_per_category}")

        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        df = pd.DataFrame(dataset)

        if categories:
            df = df[df["category"].isin(categories)]

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

        results_df = self._evaluate_mmlu(df)
        self.mmlu_results = results_df

        valid_results = results_df[results_df["success"]]
        overall_accuracy = (
            valid_results["is_correct"].mean() if not valid_results.empty else 0.0
        )

        category_accuracy = {}
        for category in valid_results["category"].unique():
            category_df = valid_results[valid_results["category"] == category]
            category_accuracy[category] = category_df["is_correct"].mean()

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

    def _process_mmlu_question(
        self, client: OpenAI, question_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single MMLU question"""
        question = question_data["question"]
        options = question_data["options"]
        correct_answer = question_data["answer"]
        category = question_data["category"]

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
            prompt = (
                "Question: "
                f"{question}\n\nOptions:\n{formatted_options}\n\n"
                "Please solve this step-by-step, then provide your final answer in the format "
                "'Answer: [letter]'."
            )
        else:
            prompt = (
                "Question: "
                f"{question}\n\nOptions:\n{formatted_options}\n\n"
                "Please choose the correct answer from the options above. Provide your answer "
                "in the format 'Answer: [letter]'."
            )

        response = client.chat.completions.create(
            model=self.model_config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.model_config.max_tokens,
            temperature=self.model_config.temperature,
        )
        response_text = response.choices[0].message.content

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

    def _extract_answer_mmlu(self, response: str) -> Optional[str]:
        """Extract MMLU answer from response"""
        match = ANSWER_PATTERN_MMLU.search(response)
        if match:
            return match.group(1).upper()

        for char in reversed(response):
            if char.upper() in "ABCDEFGHIJ":
                return char.upper()

        return None
