"""
OpenMathReasoning Dataset Implementation

NVIDIA's OpenMathReasoning dataset - high-quality math problems with detailed
chain-of-thought solutions. Contains 5.68M rows across multiple splits.

This implementation uses the 'cot' split which has 3.2M examples with detailed reasoning.
"""

import os
import random
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..dataset_interface import DatasetInfo, DatasetInterface, Question


class OpenMathReasoningDataset(DatasetInterface):
    """OpenMathReasoning dataset implementation for advanced mathematical reasoning."""

    def __init__(self):
        """Initialize OpenMathReasoning dataset."""
        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return "OpenMathReasoning"

    @property
    def supports_cot(self) -> bool:
        return True  # Has detailed chain-of-thought solutions

    def _load_raw_dataset(self, max_examples: int = 10000):
        """
        Load raw OpenMathReasoning dataset from Hugging Face.

        Args:
            max_examples: Maximum number of examples to load (default: 10000)
                         This prevents loading all 3.2M rows unnecessarily.
        """
        if self._dataset_cache is not None:
            return self._dataset_cache

        # Use STREAMING mode to avoid downloading the full 3.2M dataset
        # This way we only fetch the examples we actually need
        print(f"Loading OpenMathReasoning: {max_examples} examples (out of 3.2M total)")
        print(f"  Using streaming mode to avoid downloading full dataset...")

        dataset_stream = load_dataset(
            "nvidia/OpenMathReasoning", split="cot", streaming=True
        )

        # Take only the first max_examples from the stream
        examples = []
        for i, example in enumerate(dataset_stream):
            if i >= max_examples:
                break
            examples.append(example)
            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1}/{max_examples} examples...", end="\r")

        print(f"\n  ✓ Loaded {len(examples)} examples (streamed, not cached)")
        self._dataset_cache = pd.DataFrame(examples)
        return self._dataset_cache

    def _get_categories(self, max_examples: int = 10000) -> List[str]:
        """Get available categories in OpenMathReasoning dataset."""
        if self._categories_cache is not None:
            return self._categories_cache

        # OpenMathReasoning has problem_type and problem_source fields
        # We'll use problem_type as categories
        # Load a subset to discover categories
        df = self._load_raw_dataset(max_examples=max_examples)
        self._categories_cache = df["problem_type"].unique().tolist()
        return self._categories_cache

    def get_available_categories(self) -> List[str]:
        """Get list of all available categories in the dataset."""
        return self._get_categories()

    def load_dataset(
        self,
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42,
        max_cot_length: Optional[int] = None,
    ) -> Tuple[List[Question], DatasetInfo]:
        """
        Load OpenMathReasoning dataset with optional filtering and sampling.

        Args:
            categories: Filter by problem types
            samples_per_category: Number of samples per category
            seed: Random seed for sampling
            max_cot_length: Maximum character length for CoT solutions (for memory efficiency)
        """
        # Calculate how many examples we need to load
        # If samples_per_category is specified, we can limit loading
        # Use a buffer factor based on whether we're filtering by length
        if samples_per_category:
            # If filtering by length, load more samples to compensate
            buffer_factor = 15 if max_cot_length else 3
            estimated_needed = samples_per_category * 3 * buffer_factor
            max_to_load = min(
                estimated_needed, 100000
            )  # Cap at 100k for length filtering
        else:
            # Load more if no limit specified
            max_to_load = 50000  # Still cap to avoid loading all 3.2M

        df = self._load_raw_dataset(max_examples=max_to_load)
        available_categories = self._get_categories(max_examples=max_to_load)

        # Filter by CoT length if specified (for memory-efficient training)
        if max_cot_length:
            print(
                f"\n  📏 Filtering samples by CoT length (max: {max_cot_length} chars)"
            )
            original_count = len(df)
            df["cot_length"] = df["generated_solution"].str.len()
            df = df[df["cot_length"] <= max_cot_length]
            print(
                f"  ✓ Kept {len(df)}/{original_count} samples ({len(df)/original_count*100:.1f}%) after length filtering"
            )

            # Print distribution stats
            if len(df) > 0:
                print(f"  📊 CoT Length Stats (filtered):")
                print(f"     Min: {df['cot_length'].min()} chars")
                print(f"     Max: {df['cot_length'].max()} chars")
                print(f"     Mean: {df['cot_length'].mean():.0f} chars")
                print(f"     Median: {df['cot_length'].median():.0f} chars")

        # Filter categories if specified
        if categories:
            missing_categories = set(categories) - set(available_categories)
            if missing_categories:
                raise ValueError(
                    f"Categories not found: {missing_categories}. "
                    f"Available: {available_categories}"
                )
            df = df[df["problem_type"].isin(categories)]
            selected_categories = categories
        else:
            selected_categories = available_categories

        # Sample questions if specified (per category)
        if samples_per_category:
            np.random.seed(seed)
            random.seed(seed)

            sampled_dfs = []
            for category in selected_categories:
                category_df = df[df["problem_type"] == category]
                sample_size = min(samples_per_category, len(category_df))
                if sample_size > 0:
                    sampled_df = category_df.sample(n=sample_size, random_state=seed)
                    sampled_dfs.append(sampled_df)

            if sampled_dfs:
                df = pd.concat(sampled_dfs, ignore_index=True)
            else:
                df = pd.DataFrame()

        # Convert to Question objects
        questions = []
        for _, row in df.iterrows():
            problem_text = row["problem"]
            solution_text = row["generated_solution"]
            expected_answer = row.get("expected_answer", "")
            problem_type = row.get("problem_type", "default")

            # Clean the answer if needed
            correct_answer = str(expected_answer).strip()

            question = Question(
                question_id=f"openmr_{len(questions)}",
                question=problem_text,
                options=[],  # Free-form, no multiple choice
                correct_answer=correct_answer,
                category=problem_type,
                cot_content=solution_text,  # Full solution with detailed reasoning
                metadata={
                    "difficulty": "Advanced",
                    "type": "math_problem",
                    "problem_source": row.get("problem_source", "unknown"),
                    "generation_model": row.get("generation_model", "unknown"),
                    "pass_rate_72b_tir": row.get("pass_rate_72b_tir", "unknown"),
                },
            )
            questions.append(question)

        dataset_info = DatasetInfo(
            name="OpenMathReasoning",
            description="NVIDIA's high-quality math problems with detailed chain-of-thought reasoning",
            categories=selected_categories,
            total_questions=len(questions),
            format_type="free_form",
            difficulty_level="Advanced",
        )

        return questions, dataset_info

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format prompt for OpenMathReasoning questions."""
        if prompt_style == "plain":
            return f"""Solve this math problem:

{question.question}

Please provide your final answer in the following structured format:
The answer is [your_final_answer]

For example: The answer is 42"""

        elif prompt_style == "explicit_cot":
            return f"""Solve this math problem step by step, showing all your reasoning:

Problem: {question.question}

Please work through this step-by-step:
1. Read the problem carefully and understand what is being asked
2. Identify the given information and what needs to be found
3. Choose appropriate methods and formulas
4. Work through the solution step by step with clear explanations
5. Verify your answer makes sense
6. State your final answer clearly

Please provide your final answer in the following structured format:
The answer is [your_final_answer]

For example: The answer is 42"""

        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
