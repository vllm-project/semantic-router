"""
StrategyQA Dataset Implementation

Multi-step reasoning questions requiring implicit reasoning steps
and strategic thinking to answer yes/no questions.
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


class StrategyQADataset(DatasetInterface):
    """StrategyQA dataset implementation for multi-step implicit reasoning."""

    def __init__(self):
        """Initialize StrategyQA dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "StrategyQA"

    @property
    def supports_cot(self) -> bool:
        return True  # StrategyQA has decomposition and evidence

    def _load_raw_dataset(self):
        """Load raw StrategyQA dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Load the test split
        dataset = load_dataset("ChilleD/StrategyQA", split="test")
        self._dataset_cache = pd.DataFrame(dataset)
        return self._dataset_cache

    def _get_categories(self) -> List[str]:
        """Get available categories in StrategyQA dataset."""
        if self._categories_cache is not None:
            return self._categories_cache

        # StrategyQA doesn't have category columns, treat as single dataset
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
        """Load StrategyQA dataset with optional filtering and sampling."""
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
            answer = row["answer"]  # Boolean
            correct_answer = "Yes" if answer else "No"

            # Build CoT from decomposition and evidence if available
            cot_content = None
            if "decomposition" in row and row["decomposition"]:
                decomp = row["decomposition"]
                if isinstance(decomp, list):
                    cot_content = "Reasoning steps:\n" + "\n".join(
                        [f"{i+1}. {step}" for i, step in enumerate(decomp)]
                    )
                else:
                    cot_content = f"Reasoning: {decomp}"

            question = Question(
                question_id=f"strategyqa_{len(questions)}",
                question=question_text,
                options=["Yes", "No"],  # Binary choice
                correct_answer=correct_answer,
                category="default",
                cot_content=cot_content,
                metadata={
                    "difficulty": "Hard",
                    "type": "multi_step_reasoning",
                    "requires_implicit_steps": True,
                },
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name="StrategyQA",
            description="Multi-step reasoning questions requiring implicit reasoning steps",
            categories=selected_categories,
            total_questions=len(questions),
            format_type="binary_choice",
            difficulty_level="Hard",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format prompt for StrategyQA questions."""
        if prompt_style == "plain":
            return f"""Answer this question with Yes or No:

{question.question}

Think carefully about what information and reasoning steps are needed to answer this question.

Answer: """

        elif prompt_style == "explicit_cot":
            return f"""Answer this question with Yes or No, showing your reasoning:

Question: {question.question}

Please work through this step-by-step:
1. Break down what the question is really asking
2. Identify what facts or knowledge are needed
3. Work through the logical steps required
4. Consider any implicit assumptions or connections
5. Reach your conclusion

Show your reasoning step by step, then provide your final answer (Yes or No)."""

        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
