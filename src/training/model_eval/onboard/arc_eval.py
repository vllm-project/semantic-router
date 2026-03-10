from typing import Any, Dict, Optional

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from .constants import ANSWER_PATTERN_ARC
from .types import TestResult


class ArcEvalMixin:
    def _run_arc_challenge(self, samples: Optional[int] = 20) -> TestResult:
        """Run ARC Challenge test"""
        print(f"Starting ARC Challenge test, samples: {samples}")

        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        df = pd.DataFrame(dataset)

        if samples and len(df) > samples:
            df = df.sample(samples, random_state=self.model_config.seed)

        results_df = self._evaluate_arc(df)
        self.arc_results = results_df

        valid_results = results_df[results_df["success"]]
        overall_accuracy = (
            valid_results["is_correct"].mean() if not valid_results.empty else 0.0
        )

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

    def _process_arc_question(
        self, client: OpenAI, question_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single ARC question"""
        question = question_data["question"]
        choices = question_data["choices"]
        correct_answer = question_data["answerKey"]

        formatted_options = ""
        for label, text in zip(choices["label"], choices["text"]):
            formatted_options += f"{label}) {text}\n"

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

    def _extract_answer_arc(self, response: str) -> Optional[str]:
        """Extract ARC answer from response"""
        match = ANSWER_PATTERN_ARC.search(response)
        if match:
            return match.group(1).upper()

        for char in reversed(response):
            if char.upper() in "ABCD":
                return char.upper()

        return None
