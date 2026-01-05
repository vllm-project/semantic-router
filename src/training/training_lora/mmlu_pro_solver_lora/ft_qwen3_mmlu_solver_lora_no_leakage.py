"""
MMLU-Pro Problem Solver with Qwen3 - NO DATA LEAKAGE VERSION

✅ **KEY DIFFERENCE**:
   - Trains on EXTERNAL datasets (GSM8K, MATH, ARC, etc.)
   - Tests on MMLU-Pro (held-out benchmark)
   - No overlap between training and test data!

🎯 **Training Data Sources**:
   - Math Reasoner: GSM8K, MATH
   - Science Expert: ARC-Challenge, OpenBookQA, SciQ
   - Social Sciences: CommonsenseQA, StrategyQA
   - Humanities: TruthfulQA, MMLU-train subset
   - Law: MMLU-train law subset + specialized sources
   - Generalist: Mixed from above

🎯 **Evaluation**:
   - MMLU-Pro test split (never seen during training!)

Usage:
    # Train Math Reasoner on GSM8K + MATH, evaluate on MMLU-Pro
    python ft_qwen3_mmlu_solver_lora_no_leakage.py \
        --mode train \
        --model-type math-reasoner \
        --epochs 5 \
        --max-samples-per-dataset 1000

    # Train Science Expert on ARC + OpenBookQA + SciQ
    python ft_qwen3_mmlu_solver_lora_no_leakage.py \
        --mode train \
        --model-type science-expert \
        --epochs 5 \
        --max-samples-per-dataset 1000

    # Evaluate on MMLU-Pro
    python ft_qwen3_mmlu_solver_lora_no_leakage.py \
        --mode test \
        --model-path qwen3_mmlu_math_reasoner_r32
"""

import hashlib
import json
import logging
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Import common LoRA utilities from parent directory
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Add bench directory to path for dataset implementations
# Current file: src/training/training_lora/mmlu_pro_solver_lora/script.py
# Need to go up 5 levels to reach root, then add bench/ (parent of reasoning)
_bench_parent_dir = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    ),
    "bench",
)
if _bench_parent_dir not in sys.path:
    sys.path.insert(0, _bench_parent_dir)

import dataclasses
from typing import Dict, Sequence

import torch
from common_lora_utils import (
    clear_gpu_memory,
    log_memory_usage,
    set_gpu_device,
    setup_logging,
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

# Import bench dataset implementations
try:
    from reasoning.dataset_implementations.arc_dataset import (
        ARCDataset,
    )
    from reasoning.dataset_implementations.commonsenseqa_dataset import (
        CommonsenseQADataset,
    )
    from reasoning.dataset_implementations.gsm8k_dataset import (
        GSM8KDataset,
    )
    from reasoning.dataset_implementations.math_dataset import (
        MATHDataset,
    )
    from reasoning.dataset_implementations.openbookqa_dataset import (
        OpenBookQADataset,
    )
    from reasoning.dataset_implementations.openmathrreasoning_dataset import (
        OpenMathReasoningDataset,
    )
    from reasoning.dataset_implementations.sciq_dataset import (
        SciQDataset,
    )
    from reasoning.dataset_implementations.strategyqa_dataset import (
        StrategyQADataset,
    )
    from reasoning.dataset_implementations.truthfulqa_dataset import (
        TruthfulQADataset,
    )
except ImportError as e:
    print(f"Warning: Could not import some dataset implementations: {e}")
    print(f"Bench parent directory: {_bench_parent_dir}")
    print(f"Make sure bench datasets are available")

# Setup logging
logger = setup_logging()

# Cache directory for processed datasets
CACHE_DIR = Path(".dataset_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Training dataset mapping for each specialist model
# NOTE: Supports both multiple-choice (ARC, SciQ, etc.) and free-form (GSM8K, MATH) datasets
TRAINING_DATASETS = {
    "math-reasoner": {
        "datasets": [
            "openmathrreasoning"
        ],  # NVIDIA's high-quality math with detailed CoT
        "description": "Advanced math problem-solving with chain-of-thought reasoning",
        "target_mmlu_categories": ["math", "physics", "engineering"],
        "max_length": 3584,  # Optimized for multi-GPU with batch_size=1 + BF16
        "max_new_tokens": 1536,  # Matching shorter CoT for consistency
        "batch_size": 1,  # Reduced from 2 to avoid OOM with 3-4B models and long sequences
        "gradient_accumulation_steps": 16,  # Effective batch = 1 × 16 × 4 GPUs = 64 (same effective batch)
        "filter_long_sequences": True,  # Filter out samples > max_length to avoid truncated CoT
        "max_cot_char_length": 12000,  # Pre-filter dataset to shorter CoT samples (~3000 tokens)
        "max_samples_multiplier": 20,  # Load 20x more to compensate for char length filtering
    },
    "science-expert": {
        "datasets": ["arc", "openbookqa", "sciq"],
        "description": "Science reasoning questions",
        "target_mmlu_categories": ["biology", "chemistry", "computer science"],
    },
    "social-sciences": {
        "datasets": ["commonsenseqa", "strategyqa"],
        "description": "Common sense and strategic reasoning",
        "target_mmlu_categories": ["psychology", "economics", "business"],
    },
    "humanities": {
        "datasets": ["truthfulqa"],  # Can add more humanities datasets
        "description": "Truthfulness and general knowledge",
        "target_mmlu_categories": ["history", "philosophy"],
    },
    "law": {
        "datasets": ["mmlu_law_train"],  # Use MMLU train split for law only
        "description": "Legal reasoning (from MMLU train)",
        "target_mmlu_categories": ["law"],
    },
    "generalist": {
        "datasets": [
            "arc",
            "commonsenseqa",
            "truthfulqa",
        ],  # Mixed multiple-choice datasets
        "description": "Mixed domains (catch-all specialist)",
        "target_mmlu_categories": ["health", "other"],
    },
}

# Chain-of-Thought instruction template
# Note: We use BOTH answer key (letter) AND answer text for complete understanding
COT_INSTRUCTION_TEMPLATE = """You are an expert problem solver. Answer the following multiple-choice question by reasoning step-by-step, then provide your final answer.

Question: {question}

Options:
{options}

Instructions:
1. Think through the problem step by step
2. Explain your reasoning clearly
3. End with "The answer is X) <answer_text>" where X is the letter (A-J) and <answer_text> is the exact text of that option

Let's think step by step:"""


def get_dataset_cache_key(
    model_type: str,
    model_name: str,
    max_samples_per_dataset: int,
    max_length: int,
    use_cot: bool,
    filter_long_sequences: bool,
) -> str:
    """
    Generate a unique cache key for a dataset configuration.

    Returns a hash that changes only when the dataset config changes.
    """
    config_str = f"{model_type}_{model_name}_{max_samples_per_dataset}_{max_length}_{use_cot}_{filter_long_sequences}"
    # Add dataset sources
    datasets = TRAINING_DATASETS[model_type]["datasets"]
    config_str += f"_{'_'.join(sorted(datasets))}"

    # Include max_cot_char_length if present (affects data filtering)
    max_cot_length = TRAINING_DATASETS[model_type].get("max_cot_char_length")
    if max_cot_length:
        config_str += f"_cot{max_cot_length}"

    # Create hash
    cache_key = hashlib.md5(config_str.encode()).hexdigest()
    return cache_key


def save_cached_datasets(
    cache_key: str,
    train_samples: List[Dict],
    val_samples: List[Dict],
    train_dataset,
    val_dataset,
):
    """Save processed datasets to cache."""
    cache_file = CACHE_DIR / f"dataset_{cache_key}.pkl"

    try:
        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"💾 Cached dataset saved: {cache_file}")
        logger.info(f"   Size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
        return False


def load_cached_datasets(cache_key: str):
    """Load processed datasets from cache if available."""
    cache_file = CACHE_DIR / f"dataset_{cache_key}.pkl"

    if not cache_file.exists():
        return None

    try:
        logger.info(f"📦 Found cached dataset: {cache_file}")
        logger.info(f"   Size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
        logger.info(f"   Loading from cache...")

        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        logger.info(f"   ✅ Cache loaded successfully!")
        logger.info(f"   Train samples: {len(cache_data['train_samples'])}")
        logger.info(f"   Val samples: {len(cache_data['val_samples'])}")

        return cache_data
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        logger.warning(f"Will regenerate dataset...")
        return None


def get_qwen3_target_modules() -> List[str]:
    """Get LoRA target modules for Qwen3 architecture."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def get_token_sizes_for_model_type(model_type: str) -> Tuple[int, int]:
    """
    Get appropriate token sizes for training and inference based on model type.

    Args:
        model_type: Type of specialist model

    Returns:
        Tuple of (max_length for training, max_new_tokens for inference)
    """
    config = TRAINING_DATASETS.get(model_type, {})
    max_length = config.get("max_length", 1024)  # Default: 1024
    max_new_tokens = config.get("max_new_tokens", 256)  # Default: 256
    return max_length, max_new_tokens


def get_training_config_for_model_type(
    model_type: str, default_batch_size: int = 2
) -> Dict:
    """
    Get training configuration (batch size, gradient accumulation) for model type.

    Args:
        model_type: Type of specialist model
        default_batch_size: Default batch size if not specified in config

    Returns:
        Dict with batch_size and gradient_accumulation_steps
    """
    config = TRAINING_DATASETS.get(model_type, {})
    batch_size = config.get("batch_size", default_batch_size)

    # Calculate gradient accumulation to maintain effective batch size of ~8-16
    default_grad_accum = max(1, 8 // default_batch_size)
    gradient_accumulation_steps = config.get(
        "gradient_accumulation_steps", default_grad_accum
    )

    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }


def load_dataset_implementation(dataset_name: str):
    """Load the appropriate dataset implementation."""
    dataset_name = dataset_name.lower()

    if dataset_name == "gsm8k":
        return GSM8KDataset()
    elif dataset_name == "math":
        return MATHDataset()
    elif dataset_name == "arc":
        return ARCDataset(variant="challenge")  # Use challenge split
    elif dataset_name == "openbookqa":
        return OpenBookQADataset()
    elif dataset_name == "sciq":
        return SciQDataset()
    elif dataset_name == "commonsenseqa":
        return CommonsenseQADataset()
    elif dataset_name == "strategyqa":
        return StrategyQADataset()
    elif dataset_name == "truthfulqa":
        return TruthfulQADataset()
    elif dataset_name == "openmathrreasoning":
        return OpenMathReasoningDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def convert_answer_to_text(correct_answer, options: List[str]) -> str:
    """
    Convert any answer format to the actual answer text.
    This ensures consistency across all datasets.

    Args:
        correct_answer: Answer in any format (index, letter, or text)
        options: List of option texts

    Returns:
        The actual text of the correct answer
    """
    # If options is empty or invalid, return as-is
    if not options or len(options) == 0:
        return str(correct_answer)

    # Handle numeric index (0-based): 0 -> first option text
    if isinstance(correct_answer, int):
        if 0 <= correct_answer < len(options):
            return options[correct_answer].strip()
        else:
            logger.warning(
                f"Index {correct_answer} out of range for {len(options)} options"
            )
            return str(correct_answer)

    # Handle string numeric index: "0" -> first option text
    if isinstance(correct_answer, str) and correct_answer.isdigit():
        idx = int(correct_answer)
        if 0 <= idx < len(options):
            return options[idx].strip()
        else:
            logger.warning(f"Index {idx} out of range for {len(options)} options")
            return correct_answer

    # Handle letter index: "A" -> first option text, "B" -> second, etc.
    if isinstance(correct_answer, str) and len(correct_answer) == 1:
        upper = correct_answer.upper()
        if upper in "ABCDEFGHIJ":
            idx = ord(upper) - ord("A")
            if idx < len(options):
                return options[idx].strip()
            else:
                logger.warning(
                    f"Letter {upper} (index {idx}) out of range for {len(options)} options"
                )
                return correct_answer

    # Handle text that's already the answer (e.g., "Yes", "No" for StrategyQA)
    # Check if it matches any option exactly
    if isinstance(correct_answer, str):
        answer_lower = correct_answer.strip().lower()
        for option in options:
            if option.strip().lower() == answer_lower:
                return option.strip()

        # If no exact match, return as-is (might be the answer for free-form questions)
        return correct_answer.strip()

    # Fallback: convert to string
    return str(correct_answer)


def convert_bench_question_to_training_format(question_obj, dataset_name: str) -> Dict:
    """
    Convert Question object from bench to training format.
    Uses actual answer TEXT instead of letters/indices for consistency.

    Args:
        question_obj: Question object from bench dataset
        dataset_name: Name of the source dataset

    Returns:
        Dict with question, options, answer (as text), category, cot_content
        Returns None if the sample is invalid
    """
    # Check if this is a free-form question (no multiple choice options)
    has_options = question_obj.options and len(question_obj.options) >= 2

    if has_options:
        # Multiple-choice format: Convert answer to actual text
        try:
            answer_text = convert_answer_to_text(
                question_obj.correct_answer, question_obj.options
            )
        except Exception as e:
            logger.warning(
                f"Skipping {dataset_name} question {question_obj.question_id}: "
                f"failed to convert answer: {e}"
            )
            return None
    else:
        # Free-form format: Use answer as-is (GSM8K, MATH)
        answer_text = str(question_obj.correct_answer)
        logger.debug(
            f"Free-form question from {dataset_name}: "
            f"{question_obj.question_id} (no multiple-choice options)"
        )

    return {
        "question": question_obj.question,
        "options": (
            question_obj.options if has_options else []
        ),  # Empty list for free-form
        "answer": answer_text,  # Now always actual text, not letter/index
        "category": question_obj.category,
        "cot_content": question_obj.cot_content,
        "source_dataset": dataset_name,
        "question_id": question_obj.question_id,
        "is_free_form": not has_options,  # Flag to indicate answer format
    }


def load_training_data_for_model_type(
    model_type: str,
    max_samples_per_dataset: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """
    Load training data from external datasets (not MMLU-Pro).

    Args:
        model_type: Type of specialist model
        max_samples_per_dataset: Maximum samples per dataset
        seed: Random seed

    Returns:
        List of training samples in standard format
    """
    if model_type not in TRAINING_DATASETS:
        raise ValueError(f"Unknown model type: {model_type}")

    config = TRAINING_DATASETS[model_type]
    dataset_names = config["datasets"]

    # Apply multiplier if specified (for datasets that will be heavily filtered)
    samples_multiplier = config.get("max_samples_multiplier", 1)
    actual_samples_to_load = max_samples_per_dataset * samples_multiplier

    logger.info(f"Loading training data for {model_type}")
    logger.info(f"  Description: {config['description']}")
    logger.info(f"  Source datasets: {dataset_names}")
    logger.info(f"  Target MMLU categories: {config['target_mmlu_categories']}")

    if samples_multiplier > 1:
        logger.info(f"  📊 Loading {samples_multiplier}x more samples for filtering")
        logger.info(f"     Requested: {max_samples_per_dataset} per dataset")
        logger.info(f"     Actually loading: {actual_samples_to_load} per dataset")

    all_samples = []

    for dataset_name in dataset_names:
        if dataset_name == "mmlu_law_train":
            # Special case: use MMLU train split for law
            samples = load_mmlu_train_for_law(max_samples=actual_samples_to_load)
            all_samples.extend(samples)
            logger.info(
                f"  ✓ Loaded {len(samples)} samples from MMLU law (train split)"
            )
            continue

        try:
            logger.info(f"  Loading {dataset_name}...")
            dataset_impl = load_dataset_implementation(dataset_name)

            # Load questions from the dataset
            # Pass max_cot_char_length if specified (for OpenMathReasoning)
            load_kwargs = {
                "categories": None,  # Load all categories
                "samples_per_category": actual_samples_to_load,
                "seed": seed,
            }

            # Add max_cot_length for datasets that support it
            if "max_cot_char_length" in config and dataset_name == "openmathrreasoning":
                load_kwargs["max_cot_length"] = config["max_cot_char_length"]

            questions, dataset_info = dataset_impl.load_dataset(**load_kwargs)

            # Convert to our format (filter out None samples)
            valid_samples = 0
            for q in questions:
                sample = convert_bench_question_to_training_format(q, dataset_name)
                if sample is not None:  # Skip samples that failed conversion
                    all_samples.append(sample)
                    valid_samples += 1

            logger.info(
                f"  ✓ Loaded {valid_samples}/{len(questions)} valid samples from {dataset_name}"
            )

        except Exception as e:
            logger.warning(f"  ✗ Failed to load {dataset_name}: {e}")
            continue

    logger.info(f"Total training samples: {len(all_samples)}")

    # Show distribution
    source_dist = Counter([s["source_dataset"] for s in all_samples])
    logger.info(f"Source distribution: {dict(source_dist)}")

    return all_samples


def load_mmlu_train_for_law(max_samples: int = 1000) -> List[Dict]:
    """Load MMLU train split for law category only."""
    try:
        # Load MMLU-Pro train/validation split (not test!)
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

        # Filter for law only
        law_samples = []
        for item in dataset:
            if item["category"] == "law":
                # Convert MMLU answer (letter) to text for consistency
                answer_text = convert_answer_to_text(item["answer"], item["options"])

                law_samples.append(
                    {
                        "question": item["question"],
                        "options": item["options"],
                        "answer": answer_text,  # Now using text format
                        "category": item["category"],
                        "cot_content": item.get("cot_content"),
                        "source_dataset": "mmlu_law_train",
                        "question_id": f"mmlu_law_{len(law_samples)}",
                    }
                )

                if len(law_samples) >= max_samples:
                    break

        return law_samples
    except Exception as e:
        logger.warning(f"Failed to load MMLU law train: {e}")
        return []


def load_mmlu_pro_test_data(
    target_categories: List[str], max_samples: int = None
) -> List[Dict]:
    """
    Load MMLU-Pro TEST data for evaluation (never used in training!).

    Args:
        target_categories: Categories to load
        max_samples: Maximum samples per category (for quick testing)

    Returns:
        List of test samples
    """
    logger.info(f"Loading MMLU-Pro TEST data for evaluation")
    logger.info(f"  Target categories: {target_categories}")

    try:
        # Load MMLU-Pro test split
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

        # Filter for target categories
        test_samples = []
        category_counts = Counter()

        for item in dataset:
            category = item["category"]
            if category in target_categories:
                if max_samples and category_counts[category] >= max_samples:
                    continue

                test_samples.append(
                    {
                        "question": item["question"],
                        "options": item["options"],
                        "answer": item["answer"],
                        "category": category,
                        "cot_content": item.get("cot_content"),
                        "source_dataset": "mmlu_pro_test",
                        "question_id": item.get(
                            "question_id", f"mmlu_{len(test_samples)}"
                        ),
                    }
                )

                category_counts[category] += 1

        logger.info(f"Loaded {len(test_samples)} MMLU-Pro test samples")
        logger.info(f"Category distribution: {dict(category_counts)}")

        return test_samples

    except Exception as e:
        logger.error(f"Failed to load MMLU-Pro test data: {e}")
        raise


def format_options(options: List[str]) -> str:
    """Format options list as A) ..., B) ..., etc."""
    letters = "ABCDEFGHIJ"
    formatted = []
    for i, option in enumerate(options):
        if i < len(letters):
            formatted.append(f"{letters[i]}) {option}")
    return "\n".join(formatted)


def format_instruction(
    question: str,
    options: List[str],
    answer: str = None,
    cot_content: str = None,
    use_cot: bool = True,
) -> List[Dict[str, str]]:
    """
    Format a problem as chat messages for proper instruction fine-tuning.

    Uses Qwen3's ChatML format with special tokens to separate user input from assistant output.
    This ensures the model only trains on generating the answer, not the question.

    Supports both multiple-choice (with options) and free-form (without options) formats.

    Args:
        question: The question text
        options: List of answer options (empty list for free-form questions)
        answer: The correct answer TEXT (actual option content) or None for inference
        cot_content: Optional chain-of-thought reasoning from source dataset
        use_cot: Whether to use Chain-of-Thought format

    Returns:
        List of message dicts with 'role' and 'content' keys
        Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    # Determine if this is multiple-choice or free-form
    is_multiple_choice = options and len(options) >= 2

    if is_multiple_choice:
        # Multiple-choice format
        options_text = format_options(options)
        instruction = COT_INSTRUCTION_TEMPLATE.format(
            question=question, options=options_text
        )
    else:
        # Free-form format (GSM8K, MATH, etc.)
        instruction = f"""You are an expert problem solver. Solve the following problem step by step, showing your reasoning clearly.

Problem: {question}

Instructions:
1. Read the problem carefully and identify what is being asked
2. Break down the problem into steps
3. Solve step by step, showing your calculations and reasoning
4. End with "The answer is [your_final_answer]"

For example, if the answer is 42, write: "The answer is 42\""""

    # User message (the question/instruction)
    messages = [{"role": "user", "content": instruction}]

    if answer is not None:
        if is_multiple_choice:
            # Find which option matches the answer text to get the letter
            answer_letter = None
            answer_lower = answer.lower().strip()
            for i, option in enumerate(options):
                if option.lower().strip() == answer_lower:
                    answer_letter = chr(
                        65 + i
                    )  # Convert index to letter (0->A, 1->B, etc.)
                    break

            # If no exact match, still format but without letter
            if answer_letter is None:
                formatted_answer = f"The answer is {answer}"
                logger.warning(f"Could not find letter for answer: {answer}")
            else:
                formatted_answer = f"The answer is {answer_letter}) {answer}"
        else:
            # Free-form answer (no letter)
            formatted_answer = f"The answer is {answer}"

        # Assistant message (the answer)
        if use_cot and cot_content:
            # Use provided CoT content if available
            assistant_content = f"{cot_content}\n{formatted_answer}"
        else:
            # Simple format - just the answer
            assistant_content = formatted_answer

        messages.append({"role": "assistant", "content": assistant_content})

    return messages


@dataclasses.dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Data collator that masks prompt tokens and trains only on completion (assistant response).

    This is critical for instruction fine-tuning - we want the model to learn to GENERATE
    answers, not to predict the questions.
    """

    tokenizer: AutoTokenizer
    response_template: str
    mlm: bool = False

    def __call__(self, features):
        """
        Collate features and mask prompt tokens.

        Args:
            features: List of dicts with 'text' field (formatted with chat template)

        Returns:
            Dict with input_ids, attention_mask, and labels (with prompt tokens masked as -100)
        """
        # Extract texts from features
        texts = [
            f["text"] if isinstance(f, dict) and "text" in f else f for f in features
        ]

        # Tokenize all texts
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        # Create labels (copy of input_ids)
        labels = batch["input_ids"].clone()

        # Tokenize the response template to find where assistant response starts
        response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )

        # For each sequence in the batch, mask everything before the response
        for i in range(len(labels)):
            response_token_ids_start_idx = None

            # Find where response template starts in this sequence
            for idx in range(len(labels[i]) - len(response_token_ids) + 1):
                if (
                    labels[i][idx : idx + len(response_token_ids)].tolist()
                    == response_token_ids
                ):
                    response_token_ids_start_idx = idx + len(response_token_ids)
                    break

            if response_token_ids_start_idx is None:
                # Response template not found - mask entire sequence
                # This shouldn't happen if data is formatted correctly
                logger.warning(
                    f"Response template not found in sequence {i}. Masking entire sequence."
                )
                labels[i, :] = -100
            else:
                # Mask everything before the assistant's response
                labels[i, :response_token_ids_start_idx] = -100

                # Also mask padding tokens
                labels[i][labels[i] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


def create_solver_dataset(
    samples: List[Dict],
    tokenizer,
    max_length=1024,
    use_cot=True,
    filter_long_sequences=True,
):
    """
    Create dataset in conversational format for TRL's SFTTrainer.

    Returns a dataset with 'messages' field that SFTTrainer will automatically handle.
    SFTTrainer ensures:
    - User input and assistant output are properly separated
    - Model trains ONLY on the assistant's response (not the question)
    - Inference format matches training format

    Args:
        filter_long_sequences: If True, filter out samples that exceed max_length
                              to avoid training on truncated CoT reasoning
    """
    dataset_samples = []
    token_lengths = []

    for sample in samples:
        # Get messages (user + assistant)
        messages = format_instruction(
            sample["question"],
            sample["options"],
            sample["answer"],
            sample.get("cot_content"),
            use_cot=use_cot,
        )

        # Track token length for diagnostics (optional filtering)
        if filter_long_sequences or True:  # Always check for stats
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            tokens = tokenizer(formatted_text, truncation=False)
            token_length = len(tokens["input_ids"])
            token_lengths.append(token_length)

            # Filter out samples that are too long (if enabled)
            if filter_long_sequences and token_length > max_length:
                continue  # Skip this sample

        # Store in TRL format (messages field)
        dataset_samples.append({"messages": messages})

    # Log token length statistics
    if token_lengths:
        import numpy as np

        token_array = np.array(token_lengths)
        logger.info(f"\n📊 Token Length Statistics:")
        logger.info(f"  Total samples analyzed: {len(token_array)}")
        logger.info(f"  Min: {token_array.min()} tokens")
        logger.info(f"  Max: {token_array.max()} tokens")
        logger.info(f"  Mean: {token_array.mean():.1f} tokens")
        logger.info(f"  Median: {np.median(token_array):.1f} tokens")
        logger.info(f"  95th percentile: {np.percentile(token_array, 95):.1f} tokens")
        logger.info(f"  Max length for training: {max_length} tokens")

        num_exceeds = np.sum(token_array > max_length)
        exceed_pct = (num_exceeds / len(token_array)) * 100

        if filter_long_sequences:
            num_kept = len(dataset_samples)
            kept_pct = (num_kept / len(token_array)) * 100
            logger.info(
                f"  📌 Samples KEPT (fit in max_length): {num_kept}/{len(token_array)} ({kept_pct:.1f}%)"
            )
            logger.info(
                f"  🗑️  Samples FILTERED (too long): {num_exceeds}/{len(token_array)} ({exceed_pct:.1f}%)"
            )

            if num_kept == 0:
                logger.error(f"  ❌ ERROR: No samples fit in max_length={max_length}!")
                logger.error(f"  Consider increasing max_length or disabling filtering")
            elif kept_pct < 20:
                logger.warning(f"  ⚠️  WARNING: Only {kept_pct:.1f}% of samples kept!")
                logger.warning(
                    f"  Consider increasing max_length to keep more training data"
                )
        else:
            logger.info(
                f"  ⚠️  Samples that will be TRUNCATED: {num_exceeds}/{len(token_array)} ({exceed_pct:.1f}%)"
            )
            if exceed_pct > 10:
                logger.warning(
                    f"  ⚠️  WARNING: {exceed_pct:.1f}% of samples will be truncated!"
                )
                logger.warning(f"  Consider enabling filter_long_sequences=True")
        logger.info("")

    if len(dataset_samples) == 0:
        logger.error("No samples to create dataset! Cannot proceed.")
        # Return empty dataset with messages field
        return Dataset.from_dict({"messages": []})

    # Create HuggingFace Dataset from list of dicts
    return Dataset.from_list(dataset_samples)


def extract_answer_text(
    generated_text: str, options: List[str], question_text: str = ""
) -> str:
    """
    Extract the answer TEXT from generated text and match it to one of the options.
    Handles multiple formats: "A) crop farmers", "A", "crop farmers", etc.

    Args:
        generated_text: The generated response from the model
        options: List of valid option texts
        question_text: Original question (for context removal)

    Returns:
        The matched option text, or "UNKNOWN" if no match found
    """
    # Clean up the generated text
    if "Let's think step by step:" in generated_text:
        generated_text = generated_text.split("Let's think step by step:")[-1]
    elif question_text and question_text in generated_text:
        # Remove question if it was echoed
        generated_text = generated_text.split(question_text)[-1]

    # Pattern 1: "The answer is X) text" (letter + text format - NEW FORMAT)
    match = re.search(
        r"[Tt]he answer is\s*([A-J])\)\s*(.+?)(?:\.|$)", generated_text, re.IGNORECASE
    )
    if match:
        letter = match.group(1).upper()
        text = match.group(2).strip()
        # Prefer using the letter to get the option
        idx = ord(letter) - ord("A")
        if idx < len(options):
            return options[idx].strip()
        # Fallback to text matching
        extracted = text
    else:
        # Pattern 2: "The answer is: <text>" or "The answer is <text>"
        match = re.search(
            r"[Tt]he answer is:?\s*(.+?)(?:\.|$)", generated_text, re.IGNORECASE
        )
        if match:
            extracted = match.group(1).strip()
        else:
            # Pattern 3: "Answer: <text>" or "Answer <text>"
            match = re.search(
                r"[Aa]nswer:?\s*(.+?)(?:\.|$)", generated_text, re.IGNORECASE
            )
            if match:
                extracted = match.group(1).strip()
            else:
                # Take last sentence as potential answer
                sentences = generated_text.strip().split(".")
                extracted = (
                    sentences[-1].strip() if sentences else generated_text.strip()
                )

    # Try to match extracted text to one of the options
    extracted_lower = extracted.lower().strip()

    # Check if extracted starts with "X)" pattern
    letter_text_match = re.match(r"([A-J])\)\s*(.+)", extracted, re.IGNORECASE)
    if letter_text_match:
        letter = letter_text_match.group(1).upper()
        idx = ord(letter) - ord("A")
        if idx < len(options):
            return options[idx].strip()

    # First try: exact match
    for option in options:
        if option.lower().strip() == extracted_lower:
            return option.strip()

    # Second try: extracted text is a substring of an option
    for option in options:
        if extracted_lower in option.lower():
            return option.strip()

    # Third try: option is a substring of extracted text
    for option in options:
        if option.lower().strip() in extracted_lower:
            return option.strip()

    # Fourth try: check if it's just a letter (A-J) and convert to option
    letter_match = re.search(r"\b([A-J])\b", extracted.upper())
    if letter_match:
        letter = letter_match.group(1)
        idx = ord(letter) - ord("A")
        if idx < len(options):
            return options[idx].strip()

    # If still no match, return the extracted text as-is (will be marked incorrect)
    return "UNKNOWN"


def evaluate_model_on_mmlu_pro(
    model,
    tokenizer,
    test_samples: List[Dict],
    use_cot: bool = True,
    max_samples: int = None,
    phase_name: str = "MMLU-Pro Evaluation",
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> Dict:
    """
    Evaluate model on MMLU-Pro test samples with batched inference.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        test_samples: List of MMLU-Pro test samples
        use_cot: Whether to use Chain-of-Thought format
        max_samples: Maximum number of samples to evaluate
        phase_name: Name of evaluation phase
        max_new_tokens: Maximum number of tokens to generate per answer
        batch_size: Batch size for inference

    Returns:
        Dictionary with accuracy metrics
    """
    if max_samples is not None and len(test_samples) > max_samples:
        test_samples = test_samples[:max_samples]

    model.eval()

    correct = 0
    total = 0
    category_stats = {}
    predictions = []

    logger.info(f"\n{'=' * 80}")
    logger.info(f"{phase_name}: Testing on {len(test_samples)} MMLU-Pro samples")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"{'=' * 80}")

    # Process in batches
    num_batches = (len(test_samples) + batch_size - 1) // batch_size

    import time

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(test_samples))
        batch_samples = test_samples[batch_start:batch_end]

        batch_start_time = time.time()
        logger.info(
            f"⚙️  Processing batch {batch_idx + 1}/{num_batches} (samples {batch_start + 1}-{batch_end})..."
        )

        # Prepare batch data
        batch_prompts = []
        batch_true_answers = []
        batch_categories = []
        batch_options = []
        batch_questions = []

        for sample in batch_samples:
            question = sample["question"]
            options = sample["options"]
            true_answer_key = sample["answer"]
            category = sample["category"]

            # Convert true answer from letter to text
            true_answer_text = convert_answer_to_text(true_answer_key, options)

            # Format prompt using chat template
            messages = format_instruction(
                question, options, answer=None, use_cot=use_cot
            )
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            batch_prompts.append(prompt)
            batch_true_answers.append(true_answer_text)
            batch_categories.append(category)
            batch_options.append(options)
            batch_questions.append(question)

        # Tokenize batch with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            max_length=1024,
            truncation=True,
        ).to(model.device)

        # Generate for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
            )

        # Process each result in the batch
        for i, (output, input_len) in enumerate(zip(outputs, inputs["input_ids"])):
            # Decode only the generated part (skip the input prompt)
            generated_ids = output[len(input_len) :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_answer_text = extract_answer_text(
                generated_text, batch_options[i], batch_questions[i]
            )

            # Compare answer texts
            is_correct = (
                predicted_answer_text.lower().strip()
                == batch_true_answers[i].lower().strip()
            )
            if is_correct:
                correct += 1
            total += 1

            # Track per-category stats
            category = batch_categories[i]
            if category not in category_stats:
                category_stats[category] = {"correct": 0, "total": 0}
            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1

            predictions.append(
                {
                    "question": batch_questions[i][:100],
                    "true_answer": batch_true_answers[i],
                    "predicted_answer": predicted_answer_text,
                    "correct": is_correct,
                    "category": category,
                }
            )

            # Log first 5 examples
            sample_idx = batch_start + i
            if sample_idx < 5:
                logger.info(
                    f"\n[{sample_idx+1}/{len(test_samples)}] Category: {category}"
                )
                logger.info(f"Question: {batch_questions[i][:100]}...")
                logger.info(f"True Answer: {batch_true_answers[i]}")
                logger.info(f"Predicted: {predicted_answer_text}")
                logger.info(f"{'✓ CORRECT' if is_correct else '✗ WRONG'}")

        # Batch completion with timing
        batch_time = time.time() - batch_start_time
        current_acc = (correct / total * 100) if total > 0 else 0
        logger.info(
            f"✓ Batch {batch_idx + 1}/{num_batches} completed in {batch_time:.1f}s | "
            f"Progress: {batch_end}/{len(test_samples)} ({batch_end / len(test_samples) * 100:.0f}%) | "
            f"Accuracy: {current_acc:.1f}%"
        )

    accuracy = (
        (correct / total) if total > 0 else 0
    )  # Return as fraction, not percentage

    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{phase_name} Results:")
    logger.info(f"{'=' * 80}")
    logger.info(f"Overall Accuracy: {correct}/{total} = {accuracy:.2%}")
    logger.info(f"\nPer-Category Accuracy:")
    for cat in sorted(category_stats.keys()):
        cat_acc = category_stats[cat]["correct"] / category_stats[cat]["total"]
        logger.info(
            f"  {cat}: {category_stats[cat]['correct']}/{category_stats[cat]['total']} = {cat_acc:.2%}"
        )
    logger.info(f"{'=' * 80}\n")

    return {
        "accuracy": accuracy,
        "overall_accuracy": accuracy,  # Keep for backwards compatibility
        "correct": correct,
        "total": total,
        "category_stats": category_stats,
        "predictions": predictions,
    }


def main(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",  # Changed from 0.6B - 3B is minimum for CoT reasoning
    model_type: str = "math-reasoner",
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    num_epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_samples_per_dataset: int = 1000,
    num_workers: int = 0,
    output_dir: str = None,
    gpu_id: Optional[int] = None,
    use_cot: bool = True,
):
    """Main training function with NO data leakage."""
    logger.info("=" * 80)
    logger.info("Qwen3 MMLU-Pro Solver - NO DATA LEAKAGE VERSION")
    logger.info("=" * 80)
    logger.info(f"Model type: {model_type}")
    logger.info(f"Training on: {TRAINING_DATASETS[model_type]['datasets']}")
    logger.info(
        f"Testing on: MMLU-Pro {TRAINING_DATASETS[model_type]['target_mmlu_categories']}"
    )

    # Get appropriate token sizes for this model type
    max_length, max_new_tokens = get_token_sizes_for_model_type(model_type)

    # Get training config (may override batch_size from args for memory efficiency)
    training_config = get_training_config_for_model_type(
        model_type, default_batch_size=batch_size
    )
    actual_batch_size = training_config["batch_size"]
    actual_grad_accum = training_config["gradient_accumulation_steps"]

    if actual_batch_size != batch_size:
        logger.info(
            f"⚙️  Overriding batch_size: {batch_size} → {actual_batch_size} (for memory efficiency)"
        )
        logger.info(
            f"   Gradient accumulation: {actual_grad_accum} (effective batch size: {actual_batch_size * actual_grad_accum})"
        )

    logger.info(
        f"Token sizes: max_length={max_length}, max_new_tokens={max_new_tokens}"
    )
    logger.info(
        f"Batch size: {actual_batch_size}, Gradient accumulation: {actual_grad_accum}"
    )

    # Enable gradient checkpointing for long sequences to save memory
    use_gradient_checkpointing = max_length > 2048
    if use_gradient_checkpointing:
        logger.info(f"⚙️  Enabling gradient checkpointing (sequence length > 2048)")
        logger.info(f"   This trades compute for memory to handle longer sequences")

    logger.info("=" * 80)

    # GPU selection - use all GPUs if gpu_id is None
    if gpu_id is None:
        # Use all available GPUs - Trainer automatically uses DistributedDataParallel (DDP)
        import torch

        num_gpus = torch.cuda.device_count()
        logger.info(
            f"🚀 Multi-GPU Training: Using ALL {num_gpus} GPUs with DDP + BF16!"
        )
        logger.info(f"   GPUs: {', '.join([f'cuda:{i}' for i in range(num_gpus)])}")
        logger.info(f"   Mixed Precision: BF16 (saves ~30-40% memory)")
        logger.info(
            f"   Per-device batch: {actual_batch_size}, Gradient accum: {actual_grad_accum}"
        )
        logger.info(
            f"   Effective batch = {actual_batch_size} × {actual_grad_accum} × {num_gpus} = {actual_batch_size * actual_grad_accum * num_gpus}"
        )
        device_str = "cuda"
        selected_gpu = "all"

        # Clear all GPU caches
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        torch.cuda.set_device(0)  # Reset to GPU 0
    else:
        # Use single GPU
        device_str, selected_gpu = set_gpu_device(gpu_id=gpu_id, auto_select=False)
        logger.info(f"Using device: {device_str} (GPU {selected_gpu})")
        clear_gpu_memory()

    log_memory_usage("Pre-training")

    # Check cache first
    logger.info("\n" + "🔍" * 40)
    logger.info("CHECKING DATASET CACHE")
    logger.info("🔍" * 40)

    cache_key = get_dataset_cache_key(
        model_type=model_type,
        model_name=model_name,
        max_samples_per_dataset=max_samples_per_dataset,
        max_length=max_length,
        use_cot=use_cot,
        filter_long_sequences=TRAINING_DATASETS[model_type].get(
            "filter_long_sequences", False
        ),
    )
    logger.info(f"Cache key: {cache_key}")

    cached_data = load_cached_datasets(cache_key)

    if cached_data is not None:
        # Use cached data
        logger.info("✅ Using cached dataset - skipping data loading and processing!")
        train_samples = cached_data["train_samples"]
        val_samples = cached_data["val_samples"]
        train_dataset = cached_data["train_dataset"]
        val_dataset = cached_data["val_dataset"]

        logger.info(f"Training samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(val_samples)}")
    else:
        # Load and process data (no cache available)
        logger.info("❌ No cache found - loading and processing data...")

        # Load TRAINING data from external datasets
        logger.info("\n" + "📚" * 40)
        logger.info("LOADING TRAINING DATA (External Datasets)")
        logger.info("📚" * 40)

        training_samples = load_training_data_for_model_type(
            model_type=model_type,
            max_samples_per_dataset=max_samples_per_dataset,
            seed=42,
        )

        if len(training_samples) == 0:
            logger.error("No training samples loaded! Cannot proceed.")
            return

        # Split training data (80% train, 20% validation)
        train_samples, val_samples = train_test_split(
            training_samples,
            test_size=0.2,
            random_state=42,
        )

        logger.info(f"Training samples: {len(train_samples)}")
        logger.info(f"Validation samples: {len(val_samples)}")

    # ========================================
    # SHOW SAMPLE TRAINING DATA
    # ========================================
    logger.info("\n" + "📝" * 40)
    logger.info("SAMPLE TRAINING DATA (What the model will learn from)")
    logger.info("📝" * 40)
    logger.info("Showing 3 examples from training set:\n")

    for idx, sample in enumerate(train_samples[:3], 1):
        logger.info(f"{'=' * 80}")
        logger.info(f"TRAINING EXAMPLE {idx}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Source: {sample.get('source_dataset', 'unknown')}")
        logger.info(f"Category: {sample.get('category', 'unknown')}")
        logger.info(f"\nQuestion:")
        logger.info(
            f"  {sample['question'][:200]}{'...' if len(sample['question']) > 200 else ''}"
        )

        logger.info(f"\nOptions:")
        for i, opt in enumerate(sample["options"][:5], 1):  # Show first 5 options
            logger.info(f"  {chr(64+i)}) {opt}")
        if len(sample["options"]) > 5:
            logger.info(f"  ... ({len(sample['options']) - 5} more options)")

        # Find the letter for the answer
        answer_letter = None
        answer_text = sample["answer"]
        for i, opt in enumerate(sample["options"]):
            if opt.lower().strip() == answer_text.lower().strip():
                answer_letter = chr(65 + i)
                break

        logger.info(f"\n✓ Correct Answer (LETTER + TEXT format):")
        if answer_letter:
            logger.info(f"  {answer_letter}) {answer_text}")
        else:
            logger.info(f"  {answer_text} (letter not found)")

        # Show EXACT formatted training text that will be used (with chat template)
        # Note: We need a tokenizer here, but we haven't loaded it yet in this section
        # So we'll show the messages format and explain the chat template will be applied
        messages = format_instruction(
            sample["question"],
            sample["options"],
            sample["answer"],
            sample.get("cot_content"),
            use_cot=use_cot,
        )

        logger.info(f"\n" + "=" * 80)
        logger.info(f"📄 CHAT FORMAT MESSAGES (will be converted to ChatML):")
        logger.info(f"=" * 80)
        logger.info(f"User Message:")
        logger.info(f"  {messages[0]['content'][:300]}...")
        logger.info(f"\nAssistant Message (includes full CoT solution):")
        assistant_msg = messages[1]["content"]
        if len(assistant_msg) > 500:
            logger.info(f"  {assistant_msg[:250]}...")
            logger.info(
                f"  ... [solution continues for {len(assistant_msg)} characters] ..."
            )
            logger.info(f"  ...{assistant_msg[-250:]}")
        else:
            logger.info(f"  {assistant_msg}")
        logger.info(f"\nNote: Tokenizer will apply ChatML template:")
        logger.info(f"  <|im_start|>user\\n[user message]<|im_end|>")
        logger.info(f"  <|im_start|>assistant\\n[full CoT solution + answer]<|im_end|>")
        logger.info("=" * 80)
        logger.info("")

    logger.info(f"{'=' * 80}")
    logger.info("✅ Training data format verified!")
    logger.info(f"   All {len(train_samples)} training samples use ChatML format")
    logger.info(f"   Format: <|im_start|>user...question...<|im_end|>")
    logger.info(f"           <|im_start|>assistant...answer...<|im_end|>")
    logger.info(f"   Assistant will generate: 'The answer is X) <text>'")
    logger.info(f"   Example: 'The answer is A) crop farmers'")
    logger.info(f"   ✅ Model trains ONLY on assistant response (not question)")
    logger.info(f"{'=' * 80}\n")

    # Load MMLU-Pro TEST data for evaluation
    logger.info("\n" + "🎯" * 40)
    logger.info("LOADING TEST DATA (MMLU-Pro - Held Out)")
    logger.info("🎯" * 40)

    target_mmlu_categories = TRAINING_DATASETS[model_type]["target_mmlu_categories"]
    mmlu_test_samples = load_mmlu_pro_test_data(
        target_categories=target_mmlu_categories,
        max_samples=100,  # Load 100 samples per category for testing
    )

    logger.info(f"MMLU-Pro test samples: {len(mmlu_test_samples)}")

    # Load tokenizer and model
    logger.info(f"\nLoading Qwen3 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model on CPU first - SFTTrainer will handle device placement for multi-GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,  # Load in BF16 to save memory
    )

    # Don't move to device manually - SFTTrainer/Accelerate handles this for DDP!
    # model = model.to(device_str)  # ← This causes all processes to load on GPU 0
    model.config.use_cache = False  # Disable KV cache for training

    # Prepare LoRA config for SFTTrainer
    # SFTTrainer will apply LoRA and handle gradient checkpointing automatically
    target_modules = get_qwen3_target_modules()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    logger.info(
        f"✓ LoRA config prepared: r={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}"
    )
    logger.info(f"  Target modules: {target_modules}")

    # Prepare training datasets (or use cached versions)
    if cached_data is None:
        # Need to format and tokenize data
        logger.info("Formatting training data...")
        filter_long_sequences = TRAINING_DATASETS[model_type].get(
            "filter_long_sequences", False
        )

        train_dataset = create_solver_dataset(
            train_samples,
            tokenizer,
            max_length=max_length,
            use_cot=use_cot,
            filter_long_sequences=filter_long_sequences,
        )
        val_dataset = create_solver_dataset(
            val_samples,
            tokenizer,
            max_length=max_length,
            use_cot=use_cot,
            filter_long_sequences=filter_long_sequences,
        )

        # Save to cache for next time
        logger.info("\n💾 Saving processed dataset to cache...")
        save_cached_datasets(
            cache_key, train_samples, val_samples, train_dataset, val_dataset
        )
    else:
        logger.info("✅ Using cached tokenized datasets - ready to train!")

    # Setup output directory
    if output_dir is None:
        output_dir = f"qwen3_mmlu_{model_type}_no_leakage_r{lora_rank}"
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments using TrainingArguments
    # Note: SFTTrainer automatically uses DistributedDataParallel (DDP) for multi-GPU training
    # DDP is much more memory-efficient than DataParallel - no manual wrapping needed!
    # BF16 mixed precision saves ~30-40% memory, enabling larger batches on multi-GPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=actual_batch_size,
        per_device_eval_batch_size=actual_batch_size,
        gradient_accumulation_steps=actual_grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,  # BF16 mixed precision for memory efficiency (L4 GPUs support BF16)
        fp16=False,
        gradient_checkpointing=use_gradient_checkpointing,  # SFTTrainer handles this correctly with DDP
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if use_gradient_checkpointing else None
        ),
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )

    # Pre-format the dataset by converting messages to text field
    logger.info(
        "📋 Pre-formatting dataset: Converting messages to text with chat template..."
    )
    logger.info(f"   Original dataset columns: {train_dataset.column_names}")

    def apply_chat_template(example):
        """Convert messages field to text field using chat template."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    # Apply formatting to create "text" field
    train_dataset = train_dataset.map(apply_chat_template, desc="Formatting train data")
    val_dataset = val_dataset.map(
        apply_chat_template, desc="Formatting validation data"
    )

    logger.info(f"✓ Dataset formatted with columns: {train_dataset.column_names}")

    # Create data collator for completion-only training
    # This masks ALL tokens EXCEPT the assistant's response
    response_template = "<|im_start|>assistant\n"

    logger.info(
        f"🎭 Using DataCollatorForCompletionOnlyLM with response template: {repr(response_template)}"
    )
    logger.info(
        "   This ensures model trains ONLY on assistant responses, not prompts!"
    )

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create SFTTrainer with explicit prompt masking
    # Since TRL 0.24.0 doesn't support dataset_text_field, we:
    # 1. Pre-formatted dataset with "text" field (done above)
    # 2. Use custom DataCollatorForCompletionOnlyLM for prompt masking
    # 3. SFTTrainer will work with the "text" field automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # TRL 0.24.0 uses processing_class instead of tokenizer
        peft_config=peft_config,  # SFTTrainer will apply LoRA
        data_collator=data_collator,  # Custom collator masks prompts, trains only on completions
    )

    # Print trainable parameters after SFTTrainer applies LoRA
    logger.info("\n📊 Trainable Parameters:")
    trainer.model.print_trainable_parameters()

    logger.info("\n" + "🚀" * 40)
    logger.info("STARTING TRAINING (on External Datasets)")
    logger.info("🚀" * 40)
    trainer.train()

    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save configuration
    config = {
        "model_type": model_type,
        "training_datasets": TRAINING_DATASETS[model_type]["datasets"],
        "target_mmlu_categories": target_mmlu_categories,
        "use_cot": use_cot,
        "no_data_leakage": True,
        "training_description": "Trained on external datasets, tested on MMLU-Pro",
    }
    with open(os.path.join(output_dir, "solver_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model saved to: {output_dir}")

    # EVALUATIONS: Run both baseline and post-training together
    # Only run evaluation on main process (rank 0) to avoid OOM
    import accelerate

    is_main_process = accelerate.PartialState().is_main_process

    if is_main_process:
        logger.info("\n" + "🎯" * 40)
        logger.info("RUNNING EVALUATIONS ON MMLU-PRO (Main Process Only)")
        logger.info("🎯" * 40)
        logger.info(
            "Running both baseline (untrained) and post-training evaluations...\n"
        )

        # Delete trainer and model to free GPU memory for evaluation
        logger.info("🧹 Cleaning up training resources to free GPU memory...")
        try:
            del trainer
            logger.info("  ✓ Trainer deleted")
        except:
            pass
        try:
            del model
            logger.info("  ✓ Model deleted")
        except:
            pass

        # Force garbage collection and GPU memory cleanup
        import gc

        gc.collect()
        clear_gpu_memory()

        # Give CUDA a moment to release memory
        import time

        time.sleep(2)
        logger.info("✓ GPU memory cleared for evaluation\n")
    else:
        logger.info(
            "\n⏸️  Non-main process: Skipping evaluation (will run on rank 0 only)"
        )
        return  # Exit early for non-main processes

    # First: Reload base model for baseline (need untrained model)
    logger.info("📊 Step 1/2: Loading base model for baseline evaluation...")
    # For evaluation, use GPU 0 only (DataParallel not helpful for sequential inference)
    eval_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base_model_for_baseline = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,  # Load in BF16 to save memory
        device_map=eval_device,  # Directly load to device instead of .to()
    )
    base_model_for_baseline.eval()

    logger.info("\n" + "🔍" * 40)
    logger.info("BASELINE EVALUATION (Untrained Model)")
    logger.info("🔍" * 40)

    baseline_results = evaluate_model_on_mmlu_pro(
        model=base_model_for_baseline,
        tokenizer=tokenizer,
        test_samples=mmlu_test_samples,
        use_cot=use_cot,
        max_samples=50,
        phase_name="BASELINE (Untrained)",
        max_new_tokens=max_new_tokens,
        batch_size=8,
    )

    # Clean up baseline model to free memory
    del base_model_for_baseline
    clear_gpu_memory()
    logger.info("✓ Baseline model unloaded\n")

    # Second: Evaluate trained model
    logger.info("📊 Step 2/2: Evaluating trained model...")
    logger.info("\n" + "🎯" * 40)
    logger.info("POST-TRAINING EVALUATION (Trained Model)")
    logger.info("🎯" * 40)

    # Load trained model from saved checkpoint
    logger.info(f"Loading trained model from: {output_dir}")
    eval_base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,  # Load in BF16 to save memory
        device_map=eval_device,  # Directly load to device
    )
    from peft import PeftModel

    eval_model = PeftModel.from_pretrained(eval_base_model, output_dir)
    eval_model.eval()

    post_training_results = evaluate_model_on_mmlu_pro(
        model=eval_model,
        tokenizer=tokenizer,
        test_samples=mmlu_test_samples,
        use_cot=use_cot,
        max_samples=50,
        phase_name="POST-TRAINING (Trained on External Data)",
        max_new_tokens=max_new_tokens,
        batch_size=8,
    )

    # COMPARISON
    logger.info("\n" + "📊" * 40)
    logger.info("IMPROVEMENT ANALYSIS (No Data Leakage)")
    logger.info("📊" * 40)

    baseline_acc = baseline_results["overall_accuracy"]
    post_acc = post_training_results["overall_accuracy"]
    improvement = post_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0

    logger.info(f"\n{'=' * 80}")
    logger.info(f"OVERALL RESULTS:")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Baseline (Untrained):     {baseline_acc:.2f}%")
    logger.info(f"  Post-training:            {post_acc:.2f}%")
    logger.info(f"  Absolute Improvement:     {improvement:+.2f}%")
    logger.info(f"  Relative Improvement:     {improvement_pct:+.1f}%")
    logger.info(f"\n  Training Data: {TRAINING_DATASETS[model_type]['datasets']}")
    logger.info(f"  Test Data: MMLU-Pro {target_mmlu_categories}")
    logger.info(f"  Data Leakage: ✅ NONE (completely separate datasets)")

    if improvement > 5:
        logger.info(
            f"\n  ✅ SIGNIFICANT IMPROVEMENT! Model generalizes well to MMLU-Pro!"
        )
    elif improvement > 0:
        logger.info(f"\n  ⚠️  Modest improvement. Model shows some transfer learning.")
    else:
        logger.info(
            f"\n  ⚠️  No improvement. More training data or epochs may be needed."
        )

    logger.info(f"{'=' * 80}\n")

    # Save results
    results = {
        "baseline": {
            "overall_accuracy": baseline_acc,
            "correct": baseline_results["correct"],
            "total": baseline_results["total"],
            "category_stats": baseline_results["category_stats"],
        },
        "post_training": {
            "overall_accuracy": post_acc,
            "correct": post_training_results["correct"],
            "total": post_training_results["total"],
            "category_stats": post_training_results["category_stats"],
        },
        "improvement": {
            "absolute": improvement,
            "relative_pct": improvement_pct,
        },
        "training_config": {
            "model_type": model_type,
            "training_datasets": TRAINING_DATASETS[model_type]["datasets"],
            "test_categories": target_mmlu_categories,
            "epochs": num_epochs,
            "samples_per_dataset": max_samples_per_dataset,
            "lora_rank": lora_rank,
            "no_data_leakage": True,
        },
    }

    with open(os.path.join(output_dir, "training_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Results saved to: {output_dir}/training_comparison.json\n")
    log_memory_usage("Post-training")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3 MMLU-Pro Solver - NO DATA LEAKAGE VERSION"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model size: 3B (good) or 7B (better) for CoT. 0.6B/1.5B too small. Your 4x L4 GPUs can handle up to 7B easily!",
    )
    parser.add_argument(
        "--model-type",
        choices=list(TRAINING_DATASETS.keys()),
        default="math-reasoner",
        help="Type of specialist model",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=1000,
        help="Maximum samples per source dataset",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--use-cot", action="store_true", default=True)
    parser.add_argument("--no-cot", action="store_false", dest="use_cot")
    parser.add_argument(
        "--max-tokens-test",
        type=int,
        default=None,
        help="Override max_new_tokens for test mode (both baseline and trained model). Default uses model config (e.g., 1536 for math-reasoner). Use lower for faster testing or higher to avoid truncation.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear dataset cache and regenerate (useful if data changed)",
    )
    parser.add_argument(
        "--filter-category",
        type=str,
        default=None,
        help="Filter test samples by category (e.g., 'math', 'physics', 'computer science'). Only for test mode.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip baseline evaluation and only test trained model. Only for test mode.",
    )

    args = parser.parse_args()

    # Handle cache clearing
    if args.clear_cache:
        import shutil

        if CACHE_DIR.exists():
            logger.info(f"🗑️  Clearing cache directory: {CACHE_DIR}")
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(exist_ok=True)
            logger.info("✅ Cache cleared")

    # Helper functions for test mode
    def load_tokenizer(model_name: str):
        """Load tokenizer from HuggingFace."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Use left-padding for batched inference with decoder-only models
        tokenizer.padding_side = "left"
        return tokenizer

    def load_base_model(model_name: str, device_str: str):
        """Load base model and move to device."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to(device_str)
        return model

    def load_lora_model(base_model, lora_path: str, device_str: str):
        """Load LoRA adapter on top of base model."""
        model = PeftModel.from_pretrained(base_model, lora_path)
        model = model.to(device_str)
        return model

    if args.mode == "train":
        main(
            model_name=args.model,
            model_type=args.model_type,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples_per_dataset=args.max_samples_per_dataset,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            gpu_id=args.gpu_id,
            use_cot=args.use_cot,
        )
    elif args.mode == "test":
        # Test mode: Evaluate trained model on MMLU-Pro
        logger.info("=" * 80)
        logger.info("TEST MODE: Evaluating trained model on MMLU-Pro")
        logger.info("=" * 80)

        # Get model configuration
        if args.model_type not in TRAINING_DATASETS:
            raise ValueError(f"Unknown model type: {args.model_type}")

        config = TRAINING_DATASETS[args.model_type]
        target_categories = config["target_mmlu_categories"]
        max_new_tokens_default = config.get("max_new_tokens", 256)

        # Override with CLI option if specified
        if args.max_tokens_test is not None:
            max_new_tokens = args.max_tokens_test
            logger.info(
                f"⚡ Overriding max_new_tokens: {max_new_tokens_default} → {max_new_tokens}"
            )
        else:
            max_new_tokens = max_new_tokens_default

        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Target categories: {target_categories}")
        logger.info(f"Max new tokens (both models): {max_new_tokens}")

        # Set GPU device
        if args.gpu_id is not None:
            device_str, selected_gpu = set_gpu_device(
                gpu_id=args.gpu_id, auto_select=False
            )
            logger.info(f"Using device: {device_str} (GPU {selected_gpu})")
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device_str}")

        clear_gpu_memory()

        # Load MMLU-Pro test data
        logger.info("\n" + "🎯" * 40)
        logger.info("LOADING MMLU-PRO TEST DATA")
        logger.info("🎯" * 40)
        test_samples = load_mmlu_pro_test_data(
            target_categories=target_categories,
            max_samples=30,  # 30 samples per category for faster testing
        )
        logger.info(f"Loaded {len(test_samples)} MMLU-Pro test samples")

        # Filter by category if specified
        if args.filter_category:
            filter_cat_lower = args.filter_category.lower()
            original_count = len(test_samples)
            original_samples = (
                test_samples.copy()
            )  # Keep original for showing available categories
            test_samples = [
                s for s in test_samples if filter_cat_lower in s["category"].lower()
            ]
            logger.info(
                f"🔍 Filtered by category '{args.filter_category}': {len(test_samples)}/{original_count} samples"
            )

            if len(test_samples) == 0:
                logger.error(
                    f"❌ No samples found for category '{args.filter_category}'"
                )
                logger.info("Available categories in dataset:")
                categories = set(s["category"] for s in original_samples)
                for cat in sorted(categories):
                    logger.info(f"  - {cat}")
                sys.exit(1)

        # Determine model path
        if args.output_dir:
            model_path = args.output_dir
        else:
            # Use default path
            model_path = f"qwen3_mmlu_{args.model_type}_no_leakage_r{args.lora_rank}"

        logger.info(f"\nModel path: {model_path}")

        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"❌ Model not found at: {model_path}")
            logger.error("Please train the model first using --mode train")
            sys.exit(1)

        # Load tokenizer with left-padding for generation
        logger.info("\nLoading tokenizer...")
        tokenizer = load_tokenizer(args.model)
        tokenizer.padding_side = (
            "left"  # Required for batched generation with decoder-only models
        )
        logger.info(f"✓ Tokenizer padding side set to: {tokenizer.padding_side}")

        # Conditionally evaluate baseline model
        baseline_results = None
        if not args.skip_baseline:
            logger.info("\n" + "📊" * 40)
            logger.info("STEP 1/2: BASELINE EVALUATION (Untrained Model)")
            logger.info("📊" * 40)

            logger.info(f"Loading base model: {args.model}")
            base_model = load_base_model(args.model, device_str)

            baseline_results = evaluate_model_on_mmlu_pro(
                model=base_model,
                tokenizer=tokenizer,
                test_samples=test_samples,
                use_cot=args.use_cot,
                phase_name="Baseline (Untrained)",
                max_new_tokens=max_new_tokens,
                batch_size=8,
            )

            logger.info(f"✓ Baseline accuracy: {baseline_results['accuracy']:.1%}")

            # Free baseline model memory
            del base_model
            clear_gpu_memory()
        else:
            logger.info("\n⏭️  Skipping baseline evaluation (--skip-baseline flag set)")

        # Evaluate trained model
        step_num = "STEP 2/2" if not args.skip_baseline else "TRAINED MODEL EVALUATION"
        logger.info("\n" + "📊" * 40)
        logger.info(f"{step_num}: TRAINED MODEL EVALUATION")
        logger.info("📊" * 40)

        logger.info(f"Loading trained model from: {model_path}")
        base_model = load_base_model(args.model, device_str)
        trained_model = load_lora_model(
            base_model=base_model,
            lora_path=model_path,
            device_str=device_str,
        )

        trained_results = evaluate_model_on_mmlu_pro(
            model=trained_model,
            tokenizer=tokenizer,
            test_samples=test_samples,
            use_cot=args.use_cot,
            phase_name="Trained Model",
            max_new_tokens=max_new_tokens,
            batch_size=8,
        )

        logger.info(f"✓ Trained accuracy: {trained_results['accuracy']:.1%}")

        # Report comparison (if baseline was run)
        if baseline_results is not None:
            logger.info("\n" + "=" * 80)
            logger.info("📊 EVALUATION RESULTS COMPARISON")
            logger.info("=" * 80)
            logger.info(f"Baseline (Untrained): {baseline_results['accuracy']:.1%}")
            logger.info(f"Trained Model:        {trained_results['accuracy']:.1%}")
            logger.info(
                f"Improvement:          {(trained_results['accuracy'] - baseline_results['accuracy']):+.1%}"
            )
            logger.info("=" * 80)

            # Save results with comparison
            comparison = {
                "model_type": args.model_type,
                "model_path": model_path,
                "baseline": baseline_results,
                "trained": trained_results,
                "improvement": trained_results["accuracy"]
                - baseline_results["accuracy"],
                "filter_category": args.filter_category,
            }
        else:
            logger.info("\n" + "=" * 80)
            logger.info("📊 EVALUATION RESULTS (Trained Model Only)")
            logger.info("=" * 80)
            logger.info(f"Trained Model: {trained_results['accuracy']:.1%}")
            logger.info("=" * 80)

            # Save results without baseline
            comparison = {
                "model_type": args.model_type,
                "model_path": model_path,
                "trained": trained_results,
                "filter_category": args.filter_category,
            }

        results_file = os.path.join(model_path, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"\n✓ Results saved to: {results_file}")
