"""
Multi-Dataset Evaluation Interface

Provides abstract base classes and standardized interfaces for reasoning
dataset evaluation across MMLU, ARC, GPQA, TruthfulQA, CommonsenseQA, and HellaSwag.

Key Features:
- Unified Question and DatasetInfo data structures
- Abstract DatasetInterface for consistent implementations
- Enhanced PromptFormatter with dataset-specific optimizations
- Support for Chain-of-Thought (CoT) reasoning modes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Question:
    """
    Standardized question representation for multi-choice reasoning tasks.

    Attributes:
        question_id: Unique identifier for the question
        category: Subject or topic category
        question: The question text
        options: List of answer choices
        correct_answer: Index (int) of the correct option
        cot_content: Optional chain-of-thought reasoning
        metadata: Additional dataset-specific information
    """

    question_id: str
    category: str
    question: str
    options: List[str]
    correct_answer: str
    cot_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DatasetInfo:
    """
    Dataset metadata and configuration information.

    Attributes:
        name: Dataset name (e.g., "GPQA-Main", "ARC-Challenge")
        description: Brief description of the dataset
        categories: List of available subject categories
        total_questions: Total number of questions loaded
        format_type: Question format (typically "multiple_choice")
        difficulty_level: Complexity level (e.g., "graduate", "undergraduate")
    """

    name: str
    description: str
    categories: List[str]
    total_questions: int
    format_type: str
    difficulty_level: str


class DatasetInterface(ABC):
    """Abstract base class for all dataset implementations."""

    @abstractmethod
    def load_dataset(
        self,
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[List[Question], DatasetInfo]:
        """Load and return questions from the dataset.

        Args:
            categories: List of categories to filter by. If None, load all.
            samples_per_category: Max samples per category. If None, load all.
            seed: Random seed for reproducible sampling.

        Returns:
            Tuple of (questions_list, dataset_info)
        """
        pass

    @abstractmethod
    def get_available_categories(self) -> List[str]:
        """Get list of all available categories in the dataset."""
        pass

    @abstractmethod
    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format a question into a prompt string.

        Args:
            question: Question object to format
            prompt_style: Style of prompt ("plain", "cot", "explicit_cot")

        Returns:
            Formatted prompt string
        """
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the name of this dataset."""
        pass

    @property
    @abstractmethod
    def supports_cot(self) -> bool:
        """Return True if dataset has chain-of-thought content."""
        pass


class PromptFormatter:
    """Utility class for formatting prompts consistently across datasets."""

    @staticmethod
    def get_dataset_specific_instructions(dataset_name: str, difficulty: str) -> str:
        """Get dataset-specific instructions to improve accuracy."""
        dataset_name = dataset_name.lower()
        difficulty = difficulty.lower()

        if "gpqa" in dataset_name:
            return (
                "- This is a graduate-level scientific question\n"
                "- Consider the underlying scientific principles\n"
                "- Eliminate obviously incorrect options first\n"
            )
        elif "truthfulqa" in dataset_name:
            return (
                "- This question may contain common misconceptions\n"
                "- Be wary of answers that sound plausible but are incorrect\n"
                "- Choose the most factually accurate option\n"
            )
        elif "hellaswag" in dataset_name:
            return (
                "- Choose the most natural and logical continuation\n"
                "- Consider common sense and typical sequences of events\n"
                "- Think about what would realistically happen next\n"
            )
        elif "commonsenseqa" in dataset_name:
            return (
                "- Apply common sense reasoning\n"
                "- Consider everyday knowledge and experiences\n"
                "- Think about typical cause-and-effect relationships\n"
            )
        elif "arc" in dataset_name:
            return (
                "- This is a science question requiring logical reasoning\n"
                "- Apply scientific knowledge and principles\n"
                "- Consider the most scientifically accurate answer\n"
            )
        elif "mmlu" in dataset_name:
            return (
                "- This requires specific domain knowledge\n"
                "- Choose the most accurate and complete answer\n"
                "- Consider technical precision and accuracy\n"
            )
        else:
            return ""

    @staticmethod
    def get_letter_mapping() -> Dict[int, str]:
        """Get A-Z letter mapping for options (supports up to 26 options)."""
        return {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H",
            8: "I",
            9: "J",
            10: "K",
            11: "L",
            12: "M",
            13: "N",
            14: "O",
            15: "P",
            16: "Q",
            17: "R",
            18: "S",
            19: "T",
            20: "U",
            21: "V",
            22: "W",
            23: "X",
            24: "Y",
            25: "Z",
        }

    @staticmethod
    def format_options(options: List[str]) -> str:
        """Format options list into lettered format."""
        letter_mapping = PromptFormatter.get_letter_mapping()
        formatted = ""
        for i, option in enumerate(options):
            if option.lower() != "n/a":
                if i in letter_mapping:
                    formatted += f"{letter_mapping[i]}) {option}\n"
                else:
                    # Fallback for options beyond Z (unlikely but safe)
                    formatted += f"{i+1}.) {option}\n"
        return formatted.rstrip()

    @staticmethod
    def format_plain_prompt(question: str, options: List[str]) -> str:
        """Format a basic multiple choice prompt."""
        formatted_options = PromptFormatter.format_options(options)
        return (
            f"Question: {question}\n\nOptions:\n{formatted_options}\n\n"
            "Instructions:\n"
            "- Read the question carefully\n"
            "- Consider each option thoroughly\n"
            "- Choose the single best answer\n"
            "- Respond with ONLY the format: Answer: [letter]\n"
            "- Do not include any other text after your answer\n\n"
            "Your response:"
        )

    @staticmethod
    def format_cot_prompt(question: str, options: List[str]) -> str:
        """Format a chain-of-thought prompt."""
        formatted_options = PromptFormatter.format_options(options)
        return (
            f"Question: {question}\n\nOptions:\n{formatted_options}\n\n"
            "Instructions:\n"
            "- Think through this step-by-step\n"
            "- Analyze each option carefully\n"
            "- Explain your reasoning briefly\n"
            "- End with your final answer in the exact format: Answer: [letter]\n\n"
            "Your response:"
        )

    @staticmethod
    def format_explicit_cot_prompt(
        question: str, options: List[str], cot_content: Optional[str]
    ) -> str:
        """Format a prompt with explicit CoT content."""
        formatted_options = PromptFormatter.format_options(options)
        cot_section = f"\nExplanation: {cot_content}\n" if cot_content else "\n"
        return (
            f"Question: {question}\n\nOptions:\n{formatted_options}"
            f"{cot_section}\n"
            "Instructions:\n"
            "- Use the provided explanation as guidance\n"
            "- Consider how it applies to each option\n"
            "- Choose the best answer based on the reasoning\n"
            "- Provide your final answer in the exact format: Answer: [letter]\n\n"
            "Your response:"
        )

    @staticmethod
    def format_enhanced_prompt(
        question: str,
        options: List[str],
        dataset_name: str,
        difficulty: str,
        prompt_style: str = "plain",
    ) -> str:
        """Format an enhanced prompt with dataset-specific guidance."""
        formatted_options = PromptFormatter.format_options(options)
        dataset_instructions = PromptFormatter.get_dataset_specific_instructions(
            dataset_name, difficulty
        )

        if prompt_style == "cot":
            base_instructions = (
                "Instructions:\n"
                "- Think through this step-by-step\n"
                "- Analyze each option carefully\n"
            )
            if dataset_instructions:
                base_instructions += dataset_instructions
            base_instructions += (
                "- Explain your reasoning briefly\n"
                "- End with your final answer in the exact format: Answer: [letter]\n\n"
            )
        else:  # plain
            base_instructions = (
                "Instructions:\n"
                "- Read the question carefully\n"
                "- Consider each option thoroughly\n"
            )
            if dataset_instructions:
                base_instructions += dataset_instructions
            base_instructions += (
                "- Choose the single best answer\n"
                "- Respond with ONLY the format: Answer: [letter]\n"
                "- Do not include any other text after your answer\n\n"
            )

        return (
            f"Question: {question}\n\nOptions:\n{formatted_options}\n\n"
            f"{base_instructions}"
            "Your response:"
        )


def questions_to_dataframe(questions: List[Question]) -> pd.DataFrame:
    """Convert list of Question objects to pandas DataFrame for compatibility."""
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
        # Add metadata fields if present
        if q.metadata:
            record.update(q.metadata)
        records.append(record)
    return pd.DataFrame(records)


def dataframe_to_questions(df: pd.DataFrame) -> List[Question]:
    """Convert pandas DataFrame back to list of Question objects."""
    questions = []
    for _, row in df.iterrows():
        # Extract metadata (any columns not in the standard Question fields)
        standard_fields = {
            "question_id",
            "category",
            "question",
            "options",
            "answer",
            "cot_content",
        }
        metadata = {
            k: v for k, v in row.items() if k not in standard_fields and pd.notna(v)
        }

        question = Question(
            question_id=str(row["question_id"]),
            category=str(row["category"]),
            question=str(row["question"]),
            options=row["options"] if isinstance(row["options"], list) else [],
            correct_answer=str(row["answer"]),
            cot_content=(
                row.get("cot_content") if pd.notna(row.get("cot_content")) else None
            ),
            metadata=metadata if metadata else None,
        )
        questions.append(question)
    return questions
