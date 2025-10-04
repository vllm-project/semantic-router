"""
DROP Dataset Implementation

Discrete Reasoning Over Paragraphs - reading comprehension requiring
discrete reasoning operations over text passages.
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


class DROPDataset(DatasetInterface):
    """DROP dataset implementation for discrete reasoning over paragraphs."""

    def __init__(self):
        """Initialize DROP dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "DROP"

    @property
    def supports_cot(self) -> bool:
        return False  # DROP doesn't have built-in CoT content

    def _load_raw_dataset(self):
        """Load raw DROP dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Load the validation split (test split is not public)
        dataset = load_dataset("ucinlp/drop", split="validation")
        self._dataset_cache = pd.DataFrame(dataset)
        return self._dataset_cache

    def _get_categories(self) -> List[str]:
        """Get available categories in DROP dataset."""
        if self._categories_cache is not None:
            return self._categories_cache

        # DROP doesn't have category columns, treat as single dataset
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
        """Load DROP dataset with optional filtering and sampling."""
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
            passage = row["passage"]
            question_text = row["question"]
            # DROP has multiple possible answers
            answers_spans = row["answers_spans"]
            if answers_spans and len(answers_spans["spans"]) > 0:
                correct_answer = answers_spans["spans"][0]  # Take first valid answer
            else:
                correct_answer = "Unknown"

            # Combine passage and question
            full_question = f"Passage: {passage}\n\nQuestion: {question_text}"

            question = Question(
                question_id=f"drop_{len(questions)}",
                question=full_question,
                options=[],  # DROP is free-form, no multiple choice
                correct_answer=correct_answer,
                category="default",
                cot_content=None,
                metadata={
                    "difficulty": "Hard",
                    "type": "discrete_reasoning",
                    "passage": passage,
                    "question_only": question_text,
                },
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name="DROP",
            description="Reading comprehension requiring discrete reasoning over paragraphs",
            categories=selected_categories,
            total_questions=len(questions),
            format_type="free_form",
            difficulty_level="Hard",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format prompt for DROP questions."""
        if prompt_style == "plain":
            return f"""{question.question}

Please read the passage carefully and answer the question based on the information provided.

Please provide your answer in the following structured format:
ANSWER: [your answer]

For example: ANSWER: 68.5 or ANSWER: germans or ANSWER: Centenary Medal"""

        elif prompt_style == "explicit_cot":
            return f"""{question.question}

Please work through this step-by-step:
1. Read the passage carefully
2. Identify the key information relevant to the question
3. Determine what type of reasoning is required (counting, arithmetic, comparison, etc.)
4. Apply the necessary reasoning operations
5. Provide your final answer

Work through your reasoning step by step, then provide your final answer in the following structured format:
ANSWER: [your answer]

For example: ANSWER: 68.5 or ANSWER: germans or ANSWER: Centenary Medal"""

        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
