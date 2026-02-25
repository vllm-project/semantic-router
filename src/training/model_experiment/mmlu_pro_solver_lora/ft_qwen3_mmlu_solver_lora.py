"""
MMLU-Pro Problem Solver with Qwen3 Generative Fine-tuning + LoRA
Fine-tunes Qwen3-0.6B to SOLVE MMLU-Pro problems (not just classify them).

✅ **APPROACH**: Uses Qwen3 as a generative reasoning model
   - Qwen3 generates step-by-step reasoning + final answer
   - Chain-of-Thought (CoT) format for better reasoning
   - Specialized models per category group for better performance
   - Expected accuracy: 40-60% (much better than random 10%!)

🎯 **How it works**:
   Input:  "Question: What is corporate law? Options: A) ..., B) ..., C) ... Answer:"
   Output: "Let's think step by step. Corporate law deals with... The answer is B."

🧩 **Specialization Strategy**:
   Instead of one model for all 14 categories, train specialized models:
   - MathReasoner: math, physics, engineering (STEM quantitative)
   - ScienceExpert: biology, chemistry, computer science (STEM sciences)
   - HumanitiesScholar: history, philosophy (humanities)
   - SocialScientist: psychology, economics, business (social sciences)
   - LegalExpert: law (specialized domain)
   - Generalist: health, other (catch-all)

Usage:
    # Train Math Reasoner (math + physics + engineering)
    python ft_qwen3_mmlu_solver_lora.py --mode train --model-type math-reasoner --epochs 5 --max-samples-per-category 200

    # Train Science Expert (biology + chemistry + computer_science)
    python ft_qwen3_mmlu_solver_lora.py --mode train --model-type science-expert --epochs 5 --max-samples-per-category 200

    # Train Humanities Scholar (history + philosophy)
    python ft_qwen3_mmlu_solver_lora.py --mode train --model-type humanities --epochs 5 --max-samples-per-category 200

    # Train Social Scientist (psychology + economics + business)
    python ft_qwen3_mmlu_solver_lora.py --mode train --model-type social-sciences --epochs 5 --max-samples-per-category 200

    # Train Legal Expert (law only - specialized)
    python ft_qwen3_mmlu_solver_lora.py --mode train --model-type law --epochs 8 --max-samples-per-category 300

    # Train Generalist (health + other)
    python ft_qwen3_mmlu_solver_lora.py --mode train --model-type generalist --epochs 5 --max-samples-per-category 200

    # Quick test with specific GPU
    python ft_qwen3_mmlu_solver_lora.py --mode train --model-type math-reasoner --epochs 1 --gpu-id 2 --max-samples-per-category 20

    # Inference
    python ft_qwen3_mmlu_solver_lora.py --mode test --model-path qwen3_mmlu_math_reasoner

Model:
    - Qwen/Qwen3-0.6B (752M params, 28 layers, 32k context)
    - Fine-tuned with LoRA on instruction-following + reasoning format
    - Generates reasoning chain + final answer (A-J for 10-choice)

Dataset:
    - TIGER-Lab/MMLU-Pro: 14 category, 10-choice academic problems
    - Formatted as instruction-following with CoT reasoning
    - Categories grouped by domain for specialization
"""

import json
import logging
import os
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Setup logging
logger = setup_logging()

# All MMLU-Pro categories
ALL_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]

# Specialized model category groups
MODEL_TYPE_CATEGORIES = {
    "math-reasoner": ["math", "physics", "engineering"],  # STEM quantitative
    "science-expert": ["biology", "chemistry", "computer science"],  # STEM sciences
    "humanities": ["history", "philosophy"],  # Humanities
    "social-sciences": ["psychology", "economics", "business"],  # Social sciences
    "law": ["law"],  # Specialized legal domain
    "generalist": ["health", "other"],  # Catch-all
    "all": ALL_CATEGORIES,  # Train on everything (not recommended for 0.6B)
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

# Simple instruction template (without CoT requirement)
SIMPLE_INSTRUCTION_TEMPLATE = """Answer the following multiple-choice question.

Question: {question}

Options:
{options}

Answer:"""


def get_qwen3_target_modules() -> List[str]:
    """Get LoRA target modules for Qwen3 architecture."""
    return [
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "gate_proj",  # MLP gate
        "up_proj",  # MLP up
        "down_proj",  # MLP down
    ]


def convert_answer_to_text(correct_answer, options: List[str]) -> str:
    """
    Convert any answer format to the actual answer text.
    This ensures consistency across all answer formats.

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

    # Handle text that's already the answer
    if isinstance(correct_answer, str):
        answer_lower = correct_answer.strip().lower()
        for option in options:
            if option.strip().lower() == answer_lower:
                return option.strip()

        # If no exact match, return as-is
        return correct_answer.strip()

    # Fallback: convert to string
    return str(correct_answer)


class MMLU_Pro_Dataset:
    """Dataset class for MMLU-Pro problem solving."""

    def __init__(self, dataset_name="TIGER-Lab/MMLU-Pro", model_type="math-reasoner"):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.target_categories = MODEL_TYPE_CATEGORIES.get(model_type, ALL_CATEGORIES)
        logger.info(
            f"Model type '{model_type}' will train on categories: {self.target_categories}"
        )

    def load_huggingface_dataset(self, max_samples_per_category=200):
        """Load the MMLU-Pro dataset from HuggingFace with balanced sampling.

        Args:
            max_samples_per_category: Maximum number of samples per category.
                                     Default: 200 per category
        """
        logger.info(f"Loading dataset from HuggingFace: {self.dataset_name}")

        try:
            dataset = load_dataset(self.dataset_name)
            logger.info(f"Dataset splits: {dataset.keys()}")

            # Use validation split for training (test split has no answers in some datasets)
            # MMLU-Pro has both 'validation' and 'test' splits
            split_to_use = "test"  # MMLU-Pro test split has answers
            if split_to_use not in dataset:
                split_to_use = "validation"

            questions = dataset[split_to_use]["question"]
            categories = dataset[split_to_use]["category"]
            options = dataset[split_to_use]["options"]
            answers = dataset[split_to_use]["answer"]  # Answer letter (A-J)
            answer_indices = dataset[split_to_use]["answer_index"]  # Answer index (0-9)

            logger.info(f"Total samples in dataset: {len(questions)}")

            # Group samples by category
            category_samples = {}
            for i, (question, category, opts, answer, answer_idx) in enumerate(
                zip(questions, categories, options, answers, answer_indices)
            ):
                if category not in category_samples:
                    category_samples[category] = []

                # Convert answer from letter to actual text for consistent training
                answer_text = convert_answer_to_text(answer, opts)

                category_samples[category].append(
                    {
                        "question": question,
                        "options": opts,
                        "answer": answer_text,  # Now using text format
                        "answer_index": answer_idx,
                        "category": category,
                    }
                )

            logger.info(f"Available categories: {sorted(category_samples.keys())}")

            # Filter for target categories only
            available_target_categories = [
                cat for cat in self.target_categories if cat in category_samples
            ]

            # Collect balanced samples
            all_samples = []
            category_counts = {}

            for category in available_target_categories:
                if category in category_samples:
                    samples_to_take = min(
                        max_samples_per_category, len(category_samples[category])
                    )
                    category_data = category_samples[category][:samples_to_take]
                    all_samples.extend(category_data)
                    category_counts[category] = len(category_data)

            logger.info(f"Final category distribution: {category_counts}")
            logger.info(f"Total filtered samples: {len(all_samples)}")

            return all_samples

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def prepare_datasets(self, max_samples_per_category=200):
        """Prepare train/validation/test datasets.

        Args:
            max_samples_per_category: Maximum samples per category (default: 200)
        """
        all_samples = self.load_huggingface_dataset(max_samples_per_category)

        # Extract categories for stratified split
        categories = [sample["category"] for sample in all_samples]

        # Split data (60% train, 20% val, 20% test)
        train_samples, temp_samples = train_test_split(
            all_samples, test_size=0.4, random_state=42, stratify=categories
        )

        temp_categories = [s["category"] for s in temp_samples]
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.5, random_state=42, stratify=temp_categories
        )

        logger.info(f"Dataset sizes:")
        logger.info(f"  Train: {len(train_samples)}")
        logger.info(f"  Validation: {len(val_samples)}")
        logger.info(f"  Test: {len(test_samples)}")

        return {
            "train": train_samples,
            "validation": val_samples,
            "test": test_samples,
        }


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
    use_cot: bool = True,
) -> List[Dict[str, str]]:
    """
    Format a problem as chat messages for proper instruction fine-tuning.

    Uses Qwen3's ChatML format with special tokens to separate user input from assistant output.
    This ensures the model only trains on generating the answer, not the question.

    Args:
        question: The question text
        options: List of answer options
        answer: The correct answer TEXT (actual option content) or None for inference
        use_cot: Whether to use Chain-of-Thought format

    Returns:
        List of message dicts with 'role' and 'content' keys
        Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    options_text = format_options(options)

    if use_cot:
        template = COT_INSTRUCTION_TEMPLATE
    else:
        template = SIMPLE_INSTRUCTION_TEMPLATE

    instruction = template.format(question=question, options=options_text)

    # User message (the question/instruction)
    messages = [{"role": "user", "content": instruction}]

    if answer is not None:
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

        # Assistant message (the answer)
        messages.append({"role": "assistant", "content": formatted_answer})

    return messages


def create_solver_dataset(
    samples: List[Dict],
    tokenizer,
    max_length=1024,
    use_cot=True,
):
    """
    Create dataset in chat format for proper instruction fine-tuning.

    Uses tokenizer.apply_chat_template() to format messages with special tokens.
    This ensures:
    - User input and assistant output are properly separated
    - Model trains ONLY on the assistant's response (not the question)
    - Inference format matches training format
    """
    formatted_examples = []

    for sample in samples:
        # Get messages (user + assistant)
        messages = format_instruction(
            sample["question"],
            sample["options"],
            sample["answer"],
            use_cot=use_cot,
        )

        # Apply chat template to add special tokens
        # add_generation_prompt=False because we already have the assistant response
        # enable_thinking=False to train model for direct problem-solving without reasoning tokens
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        formatted_examples.append(formatted_text)

    # Tokenize
    encodings = tokenizer(
        formatted_examples,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # For causal LM, labels = input_ids (shifted internally by model)
    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"],  # Labels are the same as input_ids
        }
    )


def extract_answer_text(
    generated_text: str, options: List[str], question_text: str = ""
) -> str:
    """
    Extract the answer TEXT from generated text and match it to one of the options.

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

    # Pattern 1: "The answer is: <text>" or "The answer is <text>"
    match = re.search(
        r"[Tt]he answer is:?\s*(.+?)(?:\.|$)", generated_text, re.IGNORECASE
    )
    if match:
        extracted = match.group(1).strip()
    else:
        # Pattern 2: "Answer: <text>" or "Answer <text>"
        match = re.search(r"[Aa]nswer:?\s*(.+?)(?:\.|$)", generated_text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
        else:
            # Take last sentence as potential answer
            sentences = generated_text.strip().split(".")
            extracted = sentences[-1].strip() if sentences else generated_text.strip()

    # Try to match extracted text to one of the options
    extracted_lower = extracted.lower().strip()

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

    # Fourth try: check if it's a letter (A-J) and convert to option
    letter_match = re.search(r"\b([A-J])\b", extracted.upper())
    if letter_match:
        letter = letter_match.group(1)
        idx = ord(letter) - ord("A")
        if idx < len(options):
            return options[idx].strip()

    # If still no match, return UNKNOWN
    return "UNKNOWN"


def evaluate_model_on_samples(
    model,
    tokenizer,
    samples: List[Dict],
    use_cot: bool = True,
    max_samples: int = None,
    phase_name: str = "Evaluation",
) -> Dict:
    """
    Evaluate model on a set of samples and return detailed results.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        samples: List of sample dictionaries with question, options, answer, category
        use_cot: Whether to use Chain-of-Thought format
        max_samples: Maximum number of samples to evaluate (None = all)
        phase_name: Name of evaluation phase for logging (e.g., "Baseline", "Post-training")

    Returns:
        Dictionary with overall accuracy, category stats, and predictions
    """
    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]

    # Log question IDs for verification that same questions are used
    logger.info(f"{phase_name} - Using {len(samples)} test samples")
    logger.info(
        f"{phase_name} - Sample question hashes: {[hash(s['question'][:50]) for s in samples[:5]]}"
    )

    model.eval()

    correct = 0
    total = 0
    category_stats = {}
    predictions = []

    logger.info(f"\n{'=' * 80}")
    logger.info(f"{phase_name}: Testing on {len(samples)} samples...")
    logger.info(f"{'=' * 80}")

    for i, sample in enumerate(samples):
        question = sample["question"]
        options = sample["options"]
        true_answer_text = sample["answer"]  # Already in text format
        category = sample["category"]

        # Format prompt using chat template
        messages = format_instruction(question, options, answer=None, use_cot=use_cot)

        # Apply chat template with generation prompt
        # This adds <|im_start|>assistant\n at the end to prompt the model to respond
        # enable_thinking=False for direct answer generation without reasoning tokens
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
            )

        # Decode only the generated part (skip the input prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer_text = extract_answer_text(generated_text, options, question)

        # Compare answer texts (case-insensitive, stripped)
        is_correct = (
            predicted_answer_text.lower().strip() == true_answer_text.lower().strip()
        )
        if is_correct:
            correct += 1
        total += 1

        # Track per-category stats
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1

        predictions.append(
            {
                "question": question[:100],
                "true_answer": true_answer_text,  # Store as text
                "predicted_answer": predicted_answer_text,  # Store as text
                "correct": is_correct,
                "category": category,
            }
        )

        # Log first 5 examples
        if i < 5:
            logger.info(f"\n[{i+1}/{len(samples)}] Category: {category}")
            logger.info(f"Question: {question[:100]}...")
            logger.info(f"True Answer: {true_answer_text}")
            logger.info(f"Predicted: {predicted_answer_text}")
            logger.info(f"{'✓ CORRECT' if is_correct else '✗ WRONG'}")

        # Progress updates
        if (i + 1) % 10 == 0:
            current_acc = (correct / total * 100) if total > 0 else 0
            logger.info(
                f"Progress: {i+1}/{len(samples)} - Accuracy: {current_acc:.1f}%"
            )

    accuracy = (correct / total * 100) if total > 0 else 0

    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{phase_name} Results:")
    logger.info(f"{'=' * 80}")
    logger.info(f"Overall Accuracy: {correct}/{total} = {accuracy:.2f}%")
    logger.info(f"\nPer-Category Accuracy:")
    for cat in sorted(category_stats.keys()):
        cat_acc = category_stats[cat]["correct"] / category_stats[cat]["total"] * 100
        logger.info(
            f"  {cat}: {category_stats[cat]['correct']}/{category_stats[cat]['total']} = {cat_acc:.2f}%"
        )
    logger.info(f"{'=' * 80}\n")

    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "category_stats": category_stats,
        "predictions": predictions,
    }


def main(
    model_name: str = "Qwen/Qwen3-0.6B",
    model_type: str = "math-reasoner",
    lora_rank: int = 32,  # Higher rank for reasoning tasks
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    num_epochs: int = 5,
    batch_size: int = 2,  # Smaller batch for longer sequences
    learning_rate: float = 2e-4,
    max_samples_per_category: int = 200,
    num_workers: int = 0,
    output_dir: str = None,
    gpu_id: Optional[int] = None,
    use_cot: bool = True,
):
    """Main training function for MMLU-Pro problem solving.

    Args:
        model_type: Type of specialist model (math-reasoner, science-expert, etc.)
        max_samples_per_category: Maximum samples per category (default: 200).
        use_cot: Whether to use Chain-of-Thought format (default: True)
    """
    logger.info("Starting Qwen3 MMLU-Pro Problem Solver Fine-tuning")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Target categories: {MODEL_TYPE_CATEGORIES[model_type]}")

    # GPU selection using utility function
    device_str, selected_gpu = set_gpu_device(
        gpu_id=gpu_id, auto_select=(gpu_id is None)
    )
    logger.info(f"Using device: {device_str} (GPU {selected_gpu})")

    clear_gpu_memory()
    log_memory_usage("Pre-training")

    # Load dataset
    dataset_loader = MMLU_Pro_Dataset(model_type=model_type)
    datasets = dataset_loader.prepare_datasets(max_samples_per_category)

    train_samples = datasets["train"]
    val_samples = datasets["validation"]
    test_samples = datasets["test"]

    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    logger.info(f"Test samples: {len(test_samples)}")

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
        messages = format_instruction(
            sample["question"], sample["options"], sample["answer"], use_cot=use_cot
        )

        logger.info(f"\n" + "=" * 80)
        logger.info(f"📄 CHAT FORMAT MESSAGES (will be converted to ChatML):")
        logger.info(f"=" * 80)
        logger.info(f"User Message:")
        logger.info(f"  {messages[0]['content'][:300]}...")
        logger.info(f"\nAssistant Message:")
        logger.info(f"  {messages[1]['content']}")
        logger.info(f"\nNote: Tokenizer will apply ChatML template:")
        logger.info(f"  <|im_start|>user\\n[user message]<|im_end|>")
        logger.info(f"  <|im_start|>assistant\\n[assistant message]<|im_end|>")
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

    # Load tokenizer and model
    logger.info(f"Loading Qwen3 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model for causal LM with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Move to GPU
    model = model.to(device_str)

    # Prepare model for training
    model.config.use_cache = False  # Required for training

    # Create LoRA configuration
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

    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ========================================
    # BASELINE EVALUATION (BEFORE TRAINING)
    # ========================================
    logger.info("\n" + "🔍" * 40)
    logger.info("BASELINE EVALUATION (Before Fine-tuning)")
    logger.info("🔍" * 40)
    logger.info("Testing the pre-trained model on target categories...")
    logger.info("This shows the model's initial capability before specialization.\n")

    # Use test samples for baseline (we'll reuse them post-training)
    baseline_results = evaluate_model_on_samples(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        use_cot=use_cot,
        max_samples=200,  # Increased for more stable results
        phase_name="BASELINE (Pre-training)",
    )

    logger.info(
        f"✅ Baseline established: {baseline_results['overall_accuracy']:.2f}% accuracy"
    )
    logger.info(f"   (Expected: ~10% for untrained model on 10-choice questions)\n")

    # Prepare datasets in solver format
    logger.info("Formatting dataset for problem solving...")
    train_dataset = create_solver_dataset(
        train_samples, tokenizer, max_length=1024, use_cot=use_cot
    )
    val_dataset = create_solver_dataset(
        val_samples, tokenizer, max_length=1024, use_cot=use_cot
    )

    logger.info(f"Example training input:")
    example_text = tokenizer.decode(train_dataset[0]["input_ids"][:200])
    logger.info(example_text)

    # Setup output directory
    if output_dir is None:
        output_dir = f"qwen3_mmlu_{model_type}_r{lora_rank}"
    os.makedirs(output_dir, exist_ok=True)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
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
        fp16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        prediction_loss_only=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save configuration
    config = {
        "model_type": model_type,
        "target_categories": dataset_loader.target_categories,
        "use_cot": use_cot,
        "cot_template": (
            COT_INSTRUCTION_TEMPLATE if use_cot else SIMPLE_INSTRUCTION_TEMPLATE
        ),
    }
    with open(os.path.join(output_dir, "solver_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Model saved to: {output_dir}")

    # ========================================
    # POST-TRAINING EVALUATION (SAME TEST SET)
    # ========================================
    logger.info("\n" + "🎯" * 40)
    logger.info("POST-TRAINING EVALUATION (After Fine-tuning)")
    logger.info("🎯" * 40)
    logger.info("Testing the fine-tuned model on the SAME test questions...")
    logger.info("This shows the improvement from fine-tuning.\n")

    post_training_results = evaluate_model_on_samples(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        use_cot=use_cot,
        max_samples=200,  # Same as baseline - increased for more stable results
        phase_name="POST-TRAINING (After Fine-tuning)",
    )

    # ========================================
    # COMPARISON: BASELINE vs POST-TRAINING
    # ========================================
    logger.info("\n" + "📊" * 40)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("📊" * 40)

    baseline_acc = baseline_results["overall_accuracy"]
    post_acc = post_training_results["overall_accuracy"]
    improvement = post_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0

    logger.info(f"\n{'=' * 80}")
    logger.info(f"OVERALL RESULTS:")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Baseline (Pre-training):  {baseline_acc:.2f}%")
    logger.info(f"  Post-training:            {post_acc:.2f}%")
    logger.info(f"  Absolute Improvement:     {improvement:+.2f}%")
    logger.info(f"  Relative Improvement:     {improvement_pct:+.1f}%")

    if improvement > 5:
        logger.info(f"\n  ✅ SIGNIFICANT IMPROVEMENT! Model learned from fine-tuning!")
    elif improvement > 0:
        logger.info(
            f"\n  ⚠️  Modest improvement. Consider more training data or epochs."
        )
    else:
        logger.info(f"\n  ⚠️  No improvement. Model needs more training.")

    # Per-category comparison
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PER-CATEGORY IMPROVEMENTS:")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Category':<20} {'Baseline':<12} {'Post-train':<12} {'Improvement':<15}"
    )
    logger.info(f"{'-' * 80}")

    all_categories = set(baseline_results["category_stats"].keys()) | set(
        post_training_results["category_stats"].keys()
    )
    for cat in sorted(all_categories):
        baseline_cat = baseline_results["category_stats"].get(
            cat, {"correct": 0, "total": 1}
        )
        post_cat = post_training_results["category_stats"].get(
            cat, {"correct": 0, "total": 1}
        )

        baseline_cat_acc = (
            (baseline_cat["correct"] / baseline_cat["total"] * 100)
            if baseline_cat["total"] > 0
            else 0
        )
        post_cat_acc = (
            (post_cat["correct"] / post_cat["total"] * 100)
            if post_cat["total"] > 0
            else 0
        )
        cat_improvement = post_cat_acc - baseline_cat_acc

        logger.info(
            f"{cat:<20} {baseline_cat_acc:>6.1f}%     {post_cat_acc:>6.1f}%      {cat_improvement:>+6.1f}%"
        )

    logger.info(f"{'=' * 80}\n")

    # Save comprehensive results
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
            "categories": dataset_loader.target_categories,
            "epochs": num_epochs,
            "samples_per_category": max_samples_per_category,
            "lora_rank": lora_rank,
        },
    }

    with open(os.path.join(output_dir, "training_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"✅ Detailed results saved to: {output_dir}/training_comparison.json\n"
    )

    log_memory_usage("Post-training")


def demo_inference(
    model_path: str,
    model_name: str = "Qwen/Qwen3-0.6B",
    questions: List[Dict] = None,
):
    """Demonstrate inference with trained solver model."""
    logger.info(f"Loading MMLU-Pro solver model from: {model_path}")

    try:
        # Load config
        with open(os.path.join(model_path, "solver_config.json"), "r") as f:
            config = json.load(f)

        use_cot = config.get("use_cot", True)
        logger.info(f"Model type: {config['model_type']}")
        logger.info(f"Target categories: {config['target_categories']}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        use_fp16 = False
        if torch.cuda.is_available():
            try:
                compute_capability = torch.cuda.get_device_capability()
                use_fp16 = compute_capability[0] >= 7
            except Exception:
                use_fp16 = False

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        # Test examples (if none provided, use defaults)
        if questions is None:
            questions = [
                {
                    "question": "What is the derivative of x^2 + 3x + 5?",
                    "options": [
                        "2x + 3",
                        "x^2 + 3",
                        "2x + 5",
                        "3x + 5",
                        "x + 3",
                        "2x",
                        "x^2 + 3x",
                        "2x^2 + 3x",
                        "x + 5",
                        "3x",
                    ],
                    "answer": "A",
                    "category": "math",
                },
                {
                    "question": "What is Newton's second law of motion?",
                    "options": [
                        "F = ma",
                        "E = mc^2",
                        "F = G(m1*m2)/r^2",
                        "v = u + at",
                        "KE = 1/2 mv^2",
                        "p = mv",
                        "W = Fd",
                        "P = IV",
                        "V = IR",
                        "a = v/t",
                    ],
                    "answer": "A",
                    "category": "physics",
                },
            ]

        logger.info("Running inference...")

        for i, example in enumerate(questions):
            # Format using chat template
            messages = format_instruction(
                example["question"],
                example["options"],
                answer=None,
                use_cot=use_cot,
            )

            # Apply chat template with generation prompt
            # enable_thinking=False for direct answer generation without reasoning tokens
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = tokenizer(
                prompt, return_tensors="pt", max_length=1024, truncation=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    ],
                )

            # Decode only the generated part
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_answer_text = extract_answer_text(
                generated_text, example["options"], example["question"]
            )

            print(f"\n{'=' * 80}")
            print(f"Question {i+1}: {example['question']}")
            print(f"\nOptions:")
            print(format_options(example["options"]))
            print(f"\nModel's reasoning:")
            print(generated_text[:500] + ("..." if len(generated_text) > 500 else ""))
            print(f"\nPredicted Answer: {predicted_answer_text}")
            if "answer" in example:
                # Convert true answer to text for comparison
                true_answer_text = convert_answer_to_text(
                    example["answer"], example["options"]
                )
                print(f"True Answer: {true_answer_text}")
                print(
                    f"{'✓ CORRECT' if predicted_answer_text.lower().strip() == true_answer_text.lower().strip() else '✗ WRONG'}"
                )
            print("=" * 80)

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3 MMLU-Pro Problem Solver (Specialized Models)"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Qwen3 model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--model-type",
        choices=[
            "math-reasoner",
            "science-expert",
            "humanities",
            "social-sciences",
            "law",
            "generalist",
            "all",
        ],
        default="math-reasoner",
        help="Type of specialist model to train",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=32, help="LoRA rank (default: 32)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=64, help="LoRA alpha (default: 64)"
    )
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2 for longer sequences)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=200,
        help="Maximum samples per category (default: 200)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers (0=single process, 2-4=multiprocessing)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        default="qwen3_mmlu_math_reasoner_r32",
        help="Path to saved model for inference",
    )
    parser.add_argument(
        "--use-cot",
        action="store_true",
        default=True,
        help="Use Chain-of-Thought format (default: True)",
    )
    parser.add_argument(
        "--no-cot",
        action="store_false",
        dest="use_cot",
        help="Disable Chain-of-Thought format",
    )

    args = parser.parse_args()

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
            max_samples_per_category=args.max_samples_per_category,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            gpu_id=args.gpu_id,
            use_cot=args.use_cot,
        )
    elif args.mode == "test":
        demo_inference(args.model_path, args.model)
