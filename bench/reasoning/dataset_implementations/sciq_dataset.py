"""
SciQ Dataset Implementation

Science Questions - multiple choice science questions requiring
scientific reasoning and knowledge application.
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


class SciQDataset(DatasetInterface):
    """SciQ dataset implementation for scientific reasoning."""

    def __init__(self):
        """Initialize SciQ dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "SciQ"

    @property
    def supports_cot(self) -> bool:
        return False  # SciQ doesn't have built-in CoT content

    def _load_raw_dataset(self):
        """Load raw SciQ dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Load the test split
        dataset = load_dataset("sciq", split="test")
        self._dataset_cache = pd.DataFrame(dataset)
        return self._dataset_cache

    def _get_categories(self) -> List[str]:
        """Get available categories in SciQ dataset."""
        if self._categories_cache is not None:
            return self._categories_cache

        # SciQ doesn't have category columns, treat as single dataset
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
        """Load SciQ dataset with optional filtering and sampling."""
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
            correct_answer = row["correct_answer"]

            # Build options list
            options = [
                row["correct_answer"],
                row["distractor1"],
                row["distractor2"],
                row["distractor3"],
            ]
            # Shuffle options and find correct index
            random.seed(42)  # Fixed seed for reproducible option order
            shuffled_options = options.copy()
            random.shuffle(shuffled_options)
            correct_idx = shuffled_options.index(correct_answer)
            correct_letter = chr(65 + correct_idx)  # A, B, C, D

            question = Question(
                question_id=f"sciq_{len(questions)}",
                question=question_text,
                options=shuffled_options,
                correct_answer=correct_letter,
                category="default",
                cot_content=None,
                metadata={
                    "difficulty": "Moderate",
                    "type": "science_multiple_choice",
                    "support": row.get(
                        "support", ""
                    ),  # Background passage if available
                },
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name="SciQ",
            description="Science questions requiring scientific reasoning and knowledge",
            categories=selected_categories,
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="Moderate",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format prompt for SciQ questions."""
        options_text = "\n".join(
            [f"{chr(65+i)}) {opt}" for i, opt in enumerate(question.options)]
        )

        # Add support passage if available
        support_text = ""
        if question.metadata and question.metadata.get("support"):
            support_text = f"Background: {question.metadata['support']}\n\n"

        if prompt_style == "plain":
            return f"""{support_text}Question: {question.question}

{options_text}

Provide your answer in the format 'Answer: [letter]'."""

        elif prompt_style == "explicit_cot":
            return f"""{support_text}Question: {question.question}

Options:
{options_text}

Please work through this step-by-step:
1. Read the question carefully and identify what scientific concept is being tested
2. Consider any background information provided
3. Apply relevant scientific principles and knowledge
4. Eliminate incorrect options through reasoning
5. Select the best answer

Show your scientific reasoning step by step, then provide your answer in the format 'Answer: [letter]'."""

        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
