"""
MATH Dataset Implementation

Hendrycks et al. MATH dataset - 12,500 competition mathematics problems
requiring advanced mathematical reasoning across algebra, calculus, geometry, etc.
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


class MATHDataset(DatasetInterface):
    """MATH (Hendrycks et al.) dataset implementation for mathematical reasoning."""

    def __init__(self):
        """Initialize MATH dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "MATH"

    @property
    def supports_cot(self) -> bool:
        return True  # MATH has step-by-step solutions

    def _load_raw_dataset(self):
        """Load raw MATH dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Load the test split - try different possible dataset names
        try:
            dataset = load_dataset("hendrycks/math", split="test")
        except Exception:
            try:
                dataset = load_dataset("lighteval/MATH", split="test")
            except Exception:
                dataset = load_dataset("competition_math", split="test")
        self._dataset_cache = pd.DataFrame(dataset)
        return self._dataset_cache

    def _get_categories(self) -> List[str]:
        """Get available categories (subjects) in MATH dataset."""
        if self._categories_cache is not None:
            return self._categories_cache

        df = self._load_raw_dataset()
        # MATH has 'type' field for subject areas
        self._categories_cache = sorted(df["type"].unique().tolist())
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
        """Load MATH dataset with optional filtering and sampling."""
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
            df = df[df["type"].isin(categories)]
            selected_categories = categories
        else:
            selected_categories = available_categories

        # Sample questions per category
        if samples_per_category:
            sampled_dfs = []
            np.random.seed(seed)
            random.seed(seed)

            for category in selected_categories:
                category_df = df[df["type"] == category]
                if len(category_df) == 0:
                    continue

                sample_size = min(samples_per_category, len(category_df))
                sampled_df = category_df.sample(n=sample_size, random_state=seed)
                sampled_dfs.append(sampled_df)

            if sampled_dfs:
                df = pd.concat(sampled_dfs, ignore_index=True)
            else:
                df = pd.DataFrame()

        # Convert to Question objects
        questions = []
        for _, row in df.iterrows():
            # MATH problems are free-form, but we need to extract the final answer
            # The solution contains the final answer in \boxed{} format
            question_text = row["problem"]
            solution = row["solution"]

            # Extract boxed answer as the correct answer
            import re

            boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
            correct_answer = boxed_match.group(1) if boxed_match else "Unknown"

            question = Question(
                question_id=f"math_{len(questions)}",
                question=question_text,
                options=[],  # MATH is free-form, no multiple choice
                correct_answer=correct_answer,
                category=row["type"],
                cot_content=solution,  # Full solution as CoT
                metadata={
                    "level": row.get("level", "Unknown"),
                    "subject": row["type"],
                    "solution": solution,
                },
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name="MATH",
            description="Competition mathematics problems requiring advanced reasoning",
            categories=selected_categories,
            total_questions=len(questions),
            format_type="free_form",
            difficulty_level="Graduate",  # Competition math is very hard
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format prompt for MATH questions."""
        if prompt_style == "plain":
            return f"Solve this mathematics problem step by step:\n\n{question.question}\n\nProvide your final answer in the format: Answer: [your answer]"

        elif prompt_style == "explicit_cot":
            return f"""Solve this mathematics problem step by step, showing all your work:

Problem: {question.question}

Please work through this step-by-step:
1. Identify what is being asked
2. Determine the relevant mathematical concepts
3. Set up the problem
4. Solve step by step
5. Verify your answer

Provide your final answer in the format: Answer: [your answer]"""

        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
