"""
AQUA-RAT Dataset Implementation

Algebraic Question Answering with Rationales - algebraic word problems
with step-by-step rationales for mathematical reasoning evaluation.
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


class AquaRatDataset(DatasetInterface):
    """AQUA-RAT dataset implementation for algebraic reasoning with rationales."""

    def __init__(self):
        """Initialize AQUA-RAT dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "AQUA-RAT"

    @property
    def supports_cot(self) -> bool:
        return True  # AQUA-RAT has rationales

    def _load_raw_dataset(self):
        """Load raw AQUA-RAT dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Load the test split
        dataset = load_dataset("aqua_rat", split="test")
        self._dataset_cache = pd.DataFrame(dataset)
        return self._dataset_cache

    def _get_categories(self) -> List[str]:
        """Get available categories in AQUA-RAT dataset."""
        if self._categories_cache is not None:
            return self._categories_cache

        # AQUA-RAT doesn't have category columns, treat as single dataset
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
        """Load AQUA-RAT dataset with optional filtering and sampling."""
        df = self._load_raw_dataset()
        available_categories = self._get_categories()

        # Filter categories if specified
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
            raw_options = row["options"]  # List of 5 options (A, B, C, D, E)
            correct_answer = row["correct"]  # Letter (A, B, C, D, E)
            rationale = row["rationale"]  # Step-by-step explanation

            # Clean options by removing letter prefixes (e.g., "A)500" -> "500")
            options = []
            for option in raw_options:
                # Remove letter prefix like "A)", "B)", etc.
                import re

                cleaned = re.sub(r"^[A-E]\)", "", option).strip()
                options.append(cleaned)

            question = Question(
                question_id=f"aqua_rat_{len(questions)}",
                question=question_text,
                options=options,
                correct_answer=correct_answer,
                category="default",
                cot_content=rationale,
                metadata={
                    "difficulty": "Moderate",
                    "type": "algebraic_word_problem",
                    "rationale": rationale,
                },
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name="AQUA-RAT",
            description="Algebraic word problems with step-by-step rationales",
            categories=selected_categories,
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="Moderate",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format prompt for AQUA-RAT questions."""
        options_text = "\n".join(
            [f"{chr(65+i)}) {opt}" for i, opt in enumerate(question.options)]
        )

        if prompt_style == "plain":
            return f"""Solve this algebraic word problem:

{question.question}

{options_text}

Please provide your answer in the following structured format:
ANSWER: [letter]

For example: ANSWER: A"""

        elif prompt_style == "explicit_cot":
            return f"""Solve this algebraic word problem step by step:

Problem: {question.question}

Options:
{options_text}

Please work through this step-by-step:
1. Identify the variables and what is being asked
2. Set up the algebraic equations
3. Solve the equations step by step
4. Check your answer against the options
5. Select the correct answer

Please provide your final answer in the following structured format:
ANSWER: [letter]

For example: ANSWER: A"""

        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
