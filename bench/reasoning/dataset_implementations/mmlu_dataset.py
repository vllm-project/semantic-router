"""
MMLU-Pro Dataset Implementation

Academic knowledge evaluation across 14 subject categories with
Chain-of-Thought reasoning support.
"""

import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..dataset_interface import DatasetInfo, DatasetInterface, PromptFormatter, Question


class MMLUDataset(DatasetInterface):
    """MMLU-Pro dataset implementation."""

    def __init__(self):
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "MMLU-Pro"

    @property
    def supports_cot(self) -> bool:
        return True

    def load_dataset(
        self,
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[List[Question], DatasetInfo]:
        """Load MMLU-Pro dataset."""
        # Load raw dataset
        if self._dataset_cache is None:
            dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            self._dataset_cache = pd.DataFrame(dataset)

        df = self._dataset_cache.copy()
        all_categories = sorted(df["category"].unique().tolist())
        self._categories_cache = all_categories

        # Filter by categories if specified
        if categories:
            df = df[df["category"].isin(categories)]
            if df.empty:
                valid_categories = ", ".join(all_categories)
                raise ValueError(
                    f"No data found for specified categories. "
                    f"Valid categories are: {valid_categories}"
                )

        # Sample if requested
        if samples_per_category:
            random.seed(seed)
            np.random.seed(seed)
            sampled_dfs = []
            for category in df["category"].unique():
                category_df = df[df["category"] == category]
                if len(category_df) > samples_per_category:
                    sampled_df = category_df.sample(
                        samples_per_category, random_state=seed
                    )
                    sampled_dfs.append(sampled_df)
                else:
                    sampled_dfs.append(category_df)
            df = pd.concat(sampled_dfs)

        # Convert to Question objects
        questions = []
        for _, row in df.iterrows():
            question = Question(
                question_id=str(row.get("question_id", f"mmlu_{len(questions)}")),
                category=str(row["category"]),
                question=str(row["question"]),
                options=row["options"] if isinstance(row["options"], list) else [],
                correct_answer=str(row["answer"]),
                cot_content=(
                    row.get("cot_content") if pd.notna(row.get("cot_content")) else None
                ),
                metadata={
                    "source": "MMLU-Pro",
                    "difficulty": row.get("difficulty", "unknown"),
                },
            )
            questions.append(question)

        # Create dataset info
        dataset_info = DatasetInfo(
            name="MMLU-Pro",
            description="Massive Multitask Language Understanding - Professional",
            categories=list(df["category"].unique()),
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="undergraduate",
        )

        return questions, dataset_info

    def get_available_categories(self) -> List[str]:
        """Get all available MMLU categories."""
        if self._categories_cache is None:
            # Load dataset to get categories
            self.load_dataset()
        return self._categories_cache or []

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format MMLU question into prompt."""
        if prompt_style == "plain":
            return PromptFormatter.format_plain_prompt(
                question.question, question.options
            )
        elif prompt_style == "cot":
            return PromptFormatter.format_cot_prompt(
                question.question, question.options
            )
        elif prompt_style == "explicit_cot":
            return PromptFormatter.format_explicit_cot_prompt(
                question.question, question.options, question.cot_content
            )
        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")


# Legacy compatibility function
def load_mmlu_pro_dataset(
    categories: Optional[List[str]] = None,
    samples_per_category: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Legacy function for backward compatibility."""
    mmlu = MMLUDataset()
    questions, dataset_info = mmlu.load_dataset(categories, samples_per_category, seed)

    # Convert back to DataFrame format for compatibility
    records = []
    for q in questions:
        record = {
            "question_id": q.question_id,
            "category": q.category,
            "question": q.question,
            "options": q.options,
            "answer": q.correct_answer,
            "cot_content": q.cot_content,
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df, dataset_info.categories
