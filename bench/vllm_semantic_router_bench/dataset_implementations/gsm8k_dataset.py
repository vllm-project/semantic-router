"""
GSM8K Dataset Implementation

Grade School Math 8K - 8,500 elementary mathematics word problems
requiring multi-step reasoning and basic arithmetic.
"""

import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..dataset_interface import DatasetInfo, DatasetInterface, Question


class GSM8KDataset(DatasetInterface):
    """GSM8K dataset implementation for elementary mathematical reasoning."""

    def __init__(self):
        """Initialize GSM8K dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "GSM8K"

    @property
    def supports_cot(self) -> bool:
        return True  # GSM8K has step-by-step solutions

    def _load_raw_dataset(self):
        """Load raw GSM8K dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Load the test split
        dataset = load_dataset("gsm8k", "main", split="test")
        self._dataset_cache = pd.DataFrame(dataset)
        return self._dataset_cache

    def _get_categories(self) -> List[str]:
        """Get available categories in GSM8K dataset."""
        if self._categories_cache is not None:
            return self._categories_cache

        # GSM8K doesn't have category columns, treat as single dataset
        self._categories_cache = ["default"]
        return self._categories_cache

    def get_available_categories(self) -> List[str]:
        """Get list of all available categories in the dataset."""
        return self._get_categories()

    def load_dataset(
        self,
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[List[Question], DatasetInfo]:
        """Load GSM8K dataset with optional filtering and sampling."""
        df = self._load_raw_dataset()
        available_categories = self._get_categories()

        # Filter categories if specified (though GSM8K only has one category)
        if categories:
            missing_categories = set(categories) - set(available_categories)
            if missing_categories:
                raise ValueError(
                    f"Categories not found: {missing_categories}. "
                    f"Available: {available_categories}"
                )
            selected_categories = categories
        else:
            selected_categories = available_categories

        # Sample questions if specified
        if samples_per_category:
            np.random.seed(seed)
            random.seed(seed)

            sample_size = min(samples_per_category, len(df))
            df = df.sample(n=sample_size, random_state=seed)

        # Convert to Question objects
        questions = []
        for _, row in df.iterrows():
            question_text = row["question"]
            answer_text = row["answer"]

            # Extract the final numerical answer from the solution
            import re

            # GSM8K answers end with "#### [number]"
            answer_match = re.search(r"####\s*([0-9,.-]+)", answer_text)
            correct_answer = answer_match.group(1) if answer_match else "Unknown"

            question = Question(
                question_id=f"gsm8k_{len(questions)}",
                question=question_text,
                options=[],  # GSM8K is free-form, no multiple choice
                correct_answer=correct_answer,
                category="default",
                cot_content=answer_text,  # Full solution as CoT
                metadata={
                    "difficulty": "Elementary",
                    "type": "word_problem",
                    "solution": answer_text,
                },
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name="GSM8K",
            description="Grade school mathematics word problems requiring multi-step reasoning",
            categories=selected_categories,
            total_questions=len(questions),
            format_type="free_form",
            difficulty_level="Elementary",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format prompt for GSM8K questions."""
        if prompt_style == "plain":
            return f"""Solve this math word problem:

{question.question}

Please provide your final answer in the following structured format:
ANSWER: [number]

For example: ANSWER: 42"""

        elif prompt_style == "explicit_cot":
            return f"""Solve this math word problem step by step, showing all your work:

Problem: {question.question}

Please work through this step-by-step:
1. Read the problem carefully and identify what is being asked
2. Identify the given information
3. Determine what operations are needed
4. Solve step by step, showing your calculations
5. State your final answer clearly

Please provide your final answer in the following structured format:
ANSWER: [number]

For example: ANSWER: 42"""

        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
