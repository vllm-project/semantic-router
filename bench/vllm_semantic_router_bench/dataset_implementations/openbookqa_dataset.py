"""
OpenBookQA Dataset Implementation

Elementary science questions requiring reasoning over a "book" of facts.
Tests ability to combine multiple facts and apply scientific reasoning.
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


class OpenBookQADataset(DatasetInterface):
    """OpenBookQA dataset implementation for scientific reasoning with facts."""

    def __init__(self):
        """Initialize OpenBookQA dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "OpenBookQA"

    @property
    def supports_cot(self) -> bool:
        return False  # OpenBookQA doesn't have built-in CoT content

    def _load_raw_dataset(self):
        """Load raw OpenBookQA dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Load the test split
        dataset = load_dataset("openbookqa", split="test")
        self._dataset_cache = pd.DataFrame(dataset)
        return self._dataset_cache

    def _get_categories(self) -> List[str]:
        """Get available categories in OpenBookQA dataset."""
        if self._categories_cache is not None:
            return self._categories_cache

        # OpenBookQA doesn't have category columns, treat as single dataset
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
        """Load OpenBookQA dataset with optional filtering and sampling."""
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
            question_stem = row["question_stem"]
            choices = row["choices"]
            answer_key = row["answerKey"]  # A, B, C, D

            # Extract options from choices
            # Handle different possible structures for choices
            if isinstance(choices, dict) and "text" in choices:
                options = choices["text"]
            elif isinstance(choices, list):
                options = [
                    choice["text"] if isinstance(choice, dict) else choice
                    for choice in choices
                ]
            else:
                options = [str(choices)]  # Fallback

            question = Question(
                question_id=f"openbookqa_{len(questions)}",
                question=question_stem,
                options=options,
                correct_answer=answer_key,
                category="default",
                cot_content=None,
                metadata={
                    "difficulty": "Elementary",
                    "type": "science_reasoning",
                    "requires_fact_combination": True,
                },
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name="OpenBookQA",
            description="Elementary science questions requiring reasoning over scientific facts",
            categories=selected_categories,
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="Elementary",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format prompt for OpenBookQA questions."""
        options_text = "\n".join(
            [f"{chr(65+i)}) {opt}" for i, opt in enumerate(question.options)]
        )

        if prompt_style == "plain":
            return f"""Question: {question.question}

{options_text}

Think about what scientific facts and principles apply to this question.

Provide your answer in the format 'Answer: [letter]'."""

        elif prompt_style == "explicit_cot":
            return f"""Question: {question.question}

Options:
{options_text}

Please work through this step-by-step:
1. Identify what scientific concept or principle the question is testing
2. Think about relevant scientific facts that might apply
3. Consider how different facts might combine to answer the question
4. Apply scientific reasoning to eliminate incorrect options
5. Select the best answer based on scientific principles

Show your scientific reasoning step by step, then provide your answer in the format 'Answer: [letter]'."""

        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
