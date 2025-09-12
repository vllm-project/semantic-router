"""
HellaSwag dataset implementation.

This module implements the DatasetInterface for HellaSwag dataset which
tests commonsense reasoning about everyday activities and situations.
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


class HellaSwagDataset(DatasetInterface):
    """HellaSwag dataset implementation."""

    def __init__(self):
        """Initialize HellaSwag dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "HellaSwag"

    @property
    def supports_cot(self) -> bool:
        return True  # HellaSwag benefits from reasoning about context

    def _load_raw_dataset(self):
        """Load raw HellaSwag dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        try:
            # Load train and validation splits
            train_dataset = load_dataset("hellaswag", split="train")
            val_dataset = load_dataset("hellaswag", split="validation")

            # Combine both splits for more data
            train_df = pd.DataFrame(train_dataset)
            val_df = pd.DataFrame(val_dataset)
            self._dataset_cache = pd.concat([train_df, val_df], ignore_index=True)

        except Exception as e:
            print(f"Warning: Could not load HellaSwag dataset: {e}")
            print("You may need to check your internet connection or dataset access.")
            # Create empty dataframe as fallback
            self._dataset_cache = pd.DataFrame()

        return self._dataset_cache

    def _extract_categories(self, df: pd.DataFrame) -> List[str]:
        """Extract categories from HellaSwag dataset using activity labels."""
        if df.empty:
            return []

        # Use activity_label as categories, but clean them up
        def clean_activity_label(label: str) -> str:
            """Clean up activity labels to make them more readable."""
            # Remove underscores and capitalize properly
            cleaned = label.replace("_", " ").title()

            # Handle some common cases
            replacements = {
                "Tv": "TV",
                "Diy": "DIY",
                "Atv": "ATV",
                "Bmx": "BMX",
                "Sumo": "Sumo Wrestling",
                "Mma": "MMA",
            }

            for old, new in replacements.items():
                cleaned = cleaned.replace(old, new)

            return cleaned

        # Add cleaned category column
        if "category" not in df.columns:
            df["category"] = df["activity_label"].apply(clean_activity_label)

        return sorted(df["category"].unique().tolist())

    def get_available_categories(self) -> List[str]:
        """Get all available categories in the dataset."""
        if self._categories_cache is None:
            df = self._load_raw_dataset()
            self._categories_cache = self._extract_categories(df)
        return self._categories_cache

    def load_dataset(
        self,
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[List[Question], DatasetInfo]:
        """Load HellaSwag dataset with filtering and sampling."""
        df = self._load_raw_dataset()

        if df.empty:
            return [], DatasetInfo(
                name=self.dataset_name,
                categories=[],
                total_questions=0,
            )

        # Extract categories
        all_categories = self._extract_categories(df)

        # Filter by categories if specified
        if categories:
            df = df[df["category"].isin(categories)]
            if df.empty:
                valid_categories = ", ".join(all_categories)
                raise ValueError(
                    f"No data found for specified categories. Valid categories are: {valid_categories}"
                )

        # Sample questions per category if specified
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
            df = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()

        # Convert to Question objects
        questions = []
        for _, row in df.iterrows():
            # Construct the full context
            context = row["ctx"]  # This is the full context (ctx_a + ctx_b combined)
            endings = row["endings"]  # List of 4 possible endings
            correct_idx = int(str(row["label"]))  # Convert string label to int (0-3)

            question = Question(
                question_id=f"hellaswag_{row['ind']}",
                question=f"Context: {context}\n\nWhat happens next?",
                options=endings,
                correct_answer=correct_idx,  # 0-indexed
                category=row["category"],
                cot_content=None,  # HellaSwag doesn't provide CoT
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name=self.dataset_name,
            description="HellaSwag tests commonsense reasoning about everyday activities and situations",
            categories=sorted(df["category"].unique().tolist()) if not df.empty else [],
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="moderate",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, style: str = "plain") -> str:
        """Format a question into a prompt."""
        formatter = PromptFormatter()

        if style == "plain":
            return formatter.format_enhanced_prompt(
                question.question, question.options, "HellaSwag", "moderate", "plain"
            )
        elif style == "cot":
            return formatter.format_enhanced_prompt(
                question.question, question.options, "HellaSwag", "moderate", "cot"
            )
        elif style == "explicit_cot":
            return formatter.format_explicit_cot_prompt(
                question.question, question.options, question.cot_content
            )
        else:
            raise ValueError(f"Unknown prompt style: {style}")


class HellaSwagPromptFormatter(PromptFormatter):
    """Prompt formatter for HellaSwag questions."""

    def format_plain_prompt(self, question: str, options: List[str]) -> str:
        """Format a plain prompt for HellaSwag."""
        formatted_options = ""
        for i, option in enumerate(options):
            letter = chr(ord("A") + i)
            formatted_options += f"{letter}) {option}\n"

        prompt = (
            f"{question}\n\n"
            f"Options:\n{formatted_options}\n"
            f"Please choose the most logical and natural continuation. "
            f"Provide your answer in the format 'Answer: [letter]'."
        )
        return prompt

    def format_cot_prompt(self, question: str, options: List[str]) -> str:
        """Format a chain-of-thought prompt for HellaSwag."""
        formatted_options = ""
        for i, option in enumerate(options):
            letter = chr(ord("A") + i)
            formatted_options += f"{letter}) {option}\n"

        prompt = (
            f"{question}\n\n"
            f"Options:\n{formatted_options}\n"
            f"Please think step-by-step about what would most likely happen next in this situation. "
            f"Consider the context, the activity being performed, and what would be the most natural continuation. "
            f"Then provide your final answer in the format 'Answer: [letter]'."
        )
        return prompt

    def format_explicit_cot_prompt(
        self, question: str, options: List[str], cot_content: Optional[str]
    ) -> str:
        """Format an explicit chain-of-thought prompt for HellaSwag."""
        # HellaSwag doesn't provide CoT content, so fall back to regular CoT
        return self.format_cot_prompt(question, options)
