"""
CommonsenseQA dataset implementation.

This module implements the DatasetInterface for CommonsenseQA dataset which
tests commonsense reasoning across various conceptual domains.
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


class CommonsenseQADataset(DatasetInterface):
    """CommonsenseQA dataset implementation."""

    def __init__(self):
        """Initialize CommonsenseQA dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "CommonsenseQA"

    @property
    def supports_cot(self) -> bool:
        return True  # CommonsenseQA benefits from reasoning

    def _load_raw_dataset(self):
        """Load raw CommonsenseQA dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        try:
            # Load train and validation splits
            train_dataset = load_dataset("commonsense_qa", split="train")
            val_dataset = load_dataset("commonsense_qa", split="validation")

            # Combine both splits for more data
            train_df = pd.DataFrame(train_dataset)
            val_df = pd.DataFrame(val_dataset)
            self._dataset_cache = pd.concat([train_df, val_df], ignore_index=True)

        except Exception as e:
            print(f"Warning: Could not load CommonsenseQA dataset: {e}")
            print("You may need to check your internet connection or dataset access.")
            # Create empty dataframe as fallback
            self._dataset_cache = pd.DataFrame()

        return self._dataset_cache

    def _get_category(self) -> str:
        """
        CommonsenseQA doesn't have explicit subject categories.
        All questions test commonsense reasoning.
        """
        return "Common Sense"

    def get_available_categories(self) -> List[str]:
        """Get all available categories in the dataset."""
        return [self._get_category()]

    def load_dataset(
        self,
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[List[Question], DatasetInfo]:
        """Load CommonsenseQA dataset with filtering and sampling."""
        df = self._load_raw_dataset()

        if df.empty:
            return [], DatasetInfo(
                name=self.dataset_name,
                categories=[],
                total_questions=0,
            )

        # Use single category for all questions
        single_category = self._get_category()

        # Sample questions if specified (treat all questions as single category)
        if samples_per_category:
            random.seed(seed)
            np.random.seed(seed)
            if len(df) > samples_per_category:
                df = df.sample(samples_per_category, random_state=seed)

        # Convert to Question objects
        questions = []
        for _, row in df.iterrows():
            # Extract multiple choice options
            choices = row["choices"]
            choice_texts = choices["text"]
            choice_labels = choices["label"]  # ['A', 'B', 'C', 'D', 'E']

            # Find correct answer index
            answer_key = row["answerKey"]
            correct_idx = choice_labels.index(answer_key)

            question = Question(
                question_id=row["id"],
                question=row["question"],
                options=choice_texts,
                correct_answer=correct_idx,  # 0-indexed
                category=single_category,  # Use single category for all questions
                cot_content=None,  # CommonsenseQA doesn't provide CoT
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name=self.dataset_name,
            description="CommonsenseQA tests commonsense reasoning across various conceptual domains",
            categories=[single_category],  # Single category for all questions
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="hard",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, style: str = "plain") -> str:
        """Format a question into a prompt."""
        formatter = PromptFormatter()

        if style == "plain":
            return formatter.format_enhanced_prompt(
                question.question, question.options, "CommonsenseQA", "hard", "plain"
            )
        elif style == "cot":
            return formatter.format_enhanced_prompt(
                question.question, question.options, "CommonsenseQA", "hard", "cot"
            )
        elif style == "explicit_cot":
            return formatter.format_explicit_cot_prompt(
                question.question, question.options, question.cot_content
            )
        else:
            raise ValueError(f"Unknown prompt style: {style}")


class CommonsenseQAPromptFormatter(PromptFormatter):
    """Prompt formatter for CommonsenseQA questions."""

    def format_plain_prompt(self, question: str, options: List[str]) -> str:
        """Format a plain prompt for CommonsenseQA."""
        formatted_options = ""
        for i, option in enumerate(options):
            letter = chr(ord("A") + i)
            formatted_options += f"{letter}) {option}\n"

        prompt = (
            f"Question: {question}\n\n"
            f"Options:\n{formatted_options}\n"
            f"Please choose the answer that demonstrates the best commonsense reasoning. "
            f"Provide your answer in the format 'Answer: [letter]'."
        )
        return prompt

    def format_cot_prompt(self, question: str, options: List[str]) -> str:
        """Format a chain-of-thought prompt for CommonsenseQA."""
        formatted_options = ""
        for i, option in enumerate(options):
            letter = chr(ord("A") + i)
            formatted_options += f"{letter}) {option}\n"

        prompt = (
            f"Question: {question}\n\n"
            f"Options:\n{formatted_options}\n"
            f"Please think step-by-step about this question using commonsense reasoning. "
            f"Consider what you know about the world and how things typically work. "
            f"Then provide your final answer in the format 'Answer: [letter]'."
        )
        return prompt

    def format_explicit_cot_prompt(
        self, question: str, options: List[str], cot_content: Optional[str]
    ) -> str:
        """Format an explicit chain-of-thought prompt for CommonsenseQA."""
        # CommonsenseQA doesn't provide CoT content, so fall back to regular CoT
        return self.format_cot_prompt(question, options)
