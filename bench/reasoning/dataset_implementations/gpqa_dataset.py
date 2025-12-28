"""
GPQA Dataset Implementation

Graduate-level Google-proof Q&A dataset for advanced scientific reasoning
evaluation. Supports Main, Extended, and Diamond variants with Chain-of-Thought
reasoning content.
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


class GPQADataset(DatasetInterface):
    """GPQA (Graduate-level Google-proof Q&A) dataset implementation."""

    def __init__(self, subset: str = "gpqa_main"):
        """Initialize GPQA dataset.

        Args:
            subset: Which GPQA subset to use ("gpqa_main", "gpqa_extended", or "gpqa_diamond")
        """
        self.subset = subset
        valid_subsets = ["gpqa_main", "gpqa_extended", "gpqa_diamond"]
        if self.subset not in valid_subsets:
            raise ValueError(f"subset must be one of {valid_subsets}")

        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        return f"GPQA-{self.subset.replace('gpqa_', '').title()}"

    @property
    def supports_cot(self) -> bool:
        return True  # GPQA has reasoning explanations

    def _load_raw_dataset(self):
        """Load raw GPQA dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        try:
            # Try loading from the official GPQA dataset
            dataset = load_dataset("Idavidrein/gpqa", self.subset, split="train")
            self._dataset_cache = pd.DataFrame(dataset)
        except Exception as e:
            # Fallback: try alternative dataset names or warn user
            print(f"Warning: Could not load GPQA dataset {self.subset}: {e}")
            print(
                "You may need to install the dataset manually or check the dataset name."
            )
            # Create empty dataframe as fallback
            self._dataset_cache = pd.DataFrame()

        return self._dataset_cache

    def _standardize_subject_category(self, subject: str) -> str:
        """Standardize subject names to consistent categories."""
        subject_lower = subject.lower() if subject else ""

        # Map various subject names to standard categories
        if any(word in subject_lower for word in ["physics", "phys"]):
            return "Physics"
        elif any(word in subject_lower for word in ["chemistry", "chem"]):
            return "Chemistry"
        elif any(word in subject_lower for word in ["biology", "bio"]):
            return "Biology"
        elif any(word in subject_lower for word in ["math", "mathematics"]):
            return "Mathematics"
        else:
            return "Other"

    def load_dataset(
        self,
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[List[Question], DatasetInfo]:
        """Load GPQA dataset."""
        df = self._load_raw_dataset()

        if df.empty:
            # Return empty dataset if loading failed
            return [], DatasetInfo(
                name=self.dataset_name,
                description="GPQA dataset (failed to load)",
                categories=[],
                total_questions=0,
                format_type="multiple_choice",
                difficulty_level="graduate",
            )

        # Convert to Question objects
        questions = []
        for _, row in df.iterrows():
            # Handle different possible column names for GPQA
            question_text = str(row.get("Question", row.get("question", "")))

            # Extract multiple choice options
            options = []
            correct_answer = None

            # GPQA has correct answer and incorrect answers as separate columns
            correct_answer_text = None
            if "Correct Answer" in row and pd.notna(row["Correct Answer"]):
                correct_answer_text = str(row["Correct Answer"])
            elif "Answer" in row and pd.notna(row["Answer"]):
                correct_answer_text = str(row["Answer"])
            elif "answer" in row and pd.notna(row["answer"]):
                correct_answer_text = str(row["answer"])

            # Collect all answer options
            incorrect_answers = []
            for i in [1, 2, 3]:
                col_name = f"Incorrect Answer {i}"
                if col_name in row and pd.notna(row[col_name]):
                    incorrect_answers.append(str(row[col_name]))

            # Create options list with correct answer in random position
            if correct_answer_text and incorrect_answers:
                options = incorrect_answers + [correct_answer_text]
                random.shuffle(options)  # Randomize order
                correct_answer = options.index(
                    correct_answer_text
                )  # Find index after shuffle
            else:
                # Fallback: try other formats
                options = []
                correct_answer = None

                # Try to extract from individual option columns (A, B, C, D)
                for letter in ["A", "B", "C", "D"]:
                    if letter in row and pd.notna(row[letter]):
                        options.append(str(row[letter]))

                if options and correct_answer_text:
                    # Try to find correct answer in options
                    try:
                        correct_answer = options.index(correct_answer_text)
                    except ValueError:
                        correct_answer = 0  # Default to first option if not found

            # Get subject/category
            subject = row.get(
                "Subject", row.get("subject", row.get("Category", "Other"))
            )
            category = self._standardize_subject_category(str(subject))

            # Get explanation/reasoning if available
            explanation = None
            for col in ["Explanation", "explanation", "reasoning", "Reasoning"]:
                if col in row and pd.notna(row[col]):
                    explanation = str(row[col])
                    break

            # Skip questions without proper multiple choice format
            if not options or correct_answer is None:
                continue

            question = Question(
                question_id=str(row.get("Record ID", f"gpqa_{len(questions)}")),
                category=category,
                question=question_text,
                options=options,
                correct_answer=correct_answer,
                cot_content=explanation,
                metadata={
                    "source": "GPQA",
                    "subset": self.subset,
                    "difficulty": "graduate",
                    "subject": str(subject),
                },
            )
            questions.append(question)

        # Get all unique categories
        all_categories = sorted(list(set(q.category for q in questions)))
        self._categories_cache = all_categories

        # Filter by categories if specified
        if categories:
            questions = [q for q in questions if q.category in categories]
            if not questions:
                valid_categories = ", ".join(all_categories)
                raise ValueError(
                    f"No data found for specified categories. "
                    f"Valid categories are: {valid_categories}"
                )

        # Sample if requested
        if samples_per_category:
            random.seed(seed)
            np.random.seed(seed)

            # Group by category
            category_questions = {}
            for q in questions:
                if q.category not in category_questions:
                    category_questions[q.category] = []
                category_questions[q.category].append(q)

            # Sample from each category
            sampled_questions = []
            for category, cat_questions in category_questions.items():
                if len(cat_questions) > samples_per_category:
                    sampled = random.sample(cat_questions, samples_per_category)
                    sampled_questions.extend(sampled)
                else:
                    sampled_questions.extend(cat_questions)

            questions = sampled_questions

        # Create dataset info
        dataset_info = DatasetInfo(
            name=self.dataset_name,
            description="Graduate-level Google-proof Q&A benchmark",
            categories=list(set(q.category for q in questions)),
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="graduate",
        )

        return questions, dataset_info

    def get_available_categories(self) -> List[str]:
        """Get all available GPQA categories."""
        if self._categories_cache is None:
            # Load dataset to get categories
            self.load_dataset()
        return self._categories_cache or []

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format GPQA question into prompt."""
        if prompt_style == "plain":
            return PromptFormatter.format_enhanced_prompt(
                question.question, question.options, "GPQA", "graduate", "plain"
            )
        elif prompt_style == "cot":
            return PromptFormatter.format_enhanced_prompt(
                question.question, question.options, "GPQA", "graduate", "cot"
            )
        elif prompt_style == "explicit_cot":
            return PromptFormatter.format_explicit_cot_prompt(
                question.question, question.options, question.cot_content
            )
        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")


# Convenience classes for specific subsets
class GPQAMainDataset(GPQADataset):
    """GPQA Main dataset."""

    def __init__(self):
        super().__init__(subset="gpqa_main")


class GPQAExtendedDataset(GPQADataset):
    """GPQA Extended dataset."""

    def __init__(self):
        super().__init__(subset="gpqa_extended")


class GPQADiamondDataset(GPQADataset):
    """GPQA Diamond dataset (highest quality subset)."""

    def __init__(self):
        super().__init__(subset="gpqa_diamond")
