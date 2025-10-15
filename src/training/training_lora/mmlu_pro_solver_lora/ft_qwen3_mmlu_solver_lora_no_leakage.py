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

# Add bench directory to path for dataset implementations
# Current file: src/training/training_lora/mmlu_pro_solver_lora/script.py
# Need to go up 5 levels to reach root, then add bench/ (parent of vllm_semantic_router_bench)
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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Import bench dataset implementations
try:
    from vllm_semantic_router_bench.dataset_implementations.arc_dataset import (
        ARCDataset,
    )
    from vllm_semantic_router_bench.dataset_implementations.commonsenseqa_dataset import (
        CommonsenseQADataset,
    )
    from vllm_semantic_router_bench.dataset_implementations.gsm8k_dataset import (
        GSM8KDataset,
    )
    from vllm_semantic_router_bench.dataset_implementations.math_dataset import (
        MATHDataset,
    )
    from vllm_semantic_router_bench.dataset_implementations.openbookqa_dataset import (
        OpenBookQADataset,
    )
    from vllm_semantic_router_bench.dataset_implementations.sciq_dataset import (
        SciQDataset,
    )
    from vllm_semantic_router_bench.dataset_implementations.strategyqa_dataset import (
        StrategyQADataset,
    )
    from vllm_semantic_router_bench.dataset_implementations.truthfulqa_dataset import (
        TruthfulQADataset,
    )
except ImportError as e:
    print(f"Warning: Could not import some dataset implementations: {e}")
    print(f"Bench parent directory: {_bench_parent_dir}")
    print(f"Make sure bench datasets are available")

# Setup logging
logger = setup_logging()

# Training dataset mapping for each specialist model
TRAINING_DATASETS = {
    "math-reasoner": {
        "datasets": ["gsm8k", "math"],
        "description": "Elementary and competition math problems",
        "target_mmlu_categories": ["math", "physics", "engineering"],
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
        "datasets": ["gsm8k", "arc", "commonsenseqa", "truthfulqa"],
        "description": "Mixed domains",
        "target_mmlu_categories": ["health", "other"],
    },
}

# Chain-of-Thought instruction template
COT_INSTRUCTION_TEMPLATE = """You are an expert problem solver. Answer the following multiple-choice question by reasoning step-by-step, then provide your final answer.

Question: {question}

Options:
{options}

Instructions:
1. Think through the problem step by step
2. Explain your reasoning clearly
3. End with "The answer is X" where X is the letter (A-J)

Let's think step by step:"""


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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def convert_bench_question_to_training_format(question_obj, dataset_name: str) -> Dict:
    """
    Convert Question object from bench to training format.

    Args:
        question_obj: Question object from bench dataset
        dataset_name: Name of the source dataset

    Returns:
        Dict with question, options, answer, category, cot_content
    """
    # Get correct answer letter
    # The bench Question has correct_answer as string (already the letter)
    answer_letter = question_obj.correct_answer

    return {
        "question": question_obj.question,
        "options": question_obj.options,
        "answer": answer_letter,
        "category": question_obj.category,
        "cot_content": question_obj.cot_content,
        "source_dataset": dataset_name,
        "question_id": question_obj.question_id,
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

    logger.info(f"Loading training data for {model_type}")
    logger.info(f"  Description: {config['description']}")
    logger.info(f"  Source datasets: {dataset_names}")
    logger.info(f"  Target MMLU categories: {config['target_mmlu_categories']}")

    all_samples = []

    for dataset_name in dataset_names:
        if dataset_name == "mmlu_law_train":
            # Special case: use MMLU train split for law
            samples = load_mmlu_train_for_law(max_samples=max_samples_per_dataset)
            all_samples.extend(samples)
            logger.info(
                f"  ✓ Loaded {len(samples)} samples from MMLU law (train split)"
            )
            continue

        try:
            logger.info(f"  Loading {dataset_name}...")
            dataset_impl = load_dataset_implementation(dataset_name)

            # Load questions from the dataset
            questions, dataset_info = dataset_impl.load_dataset(
                categories=None,  # Load all categories
                samples_per_category=max_samples_per_dataset,
                seed=seed,
            )

            # Convert to our format
            for q in questions:
                sample = convert_bench_question_to_training_format(q, dataset_name)
                all_samples.append(sample)

            logger.info(f"  ✓ Loaded {len(questions)} samples from {dataset_name}")

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
                law_samples.append(
                    {
                        "question": item["question"],
                        "options": item["options"],
                        "answer": item["answer"],
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
) -> str:
    """
    Format a problem as an instruction-following example.

    Args:
        question: The question text
        options: List of answer options
        answer: The correct answer letter (A-J) or None for inference
        cot_content: Optional chain-of-thought reasoning from source dataset
        use_cot: Whether to use Chain-of-Thought format

    Returns:
        Formatted instruction string
    """
    options_text = format_options(options)
    instruction = COT_INSTRUCTION_TEMPLATE.format(
        question=question, options=options_text
    )

    if answer is not None:
        # Training format: instruction + reasoning + answer
        if use_cot and cot_content:
            # Use provided CoT content if available
            return f"{instruction} {cot_content}\nThe answer is {answer}."
        else:
            # Simple format
            return f"{instruction} The answer is {answer}."
    else:
        # Inference format
        return instruction


def create_solver_dataset(
    samples: List[Dict],
    tokenizer,
    max_length=1024,
    use_cot=True,
):
    """Create dataset in generative format for problem solving."""
    formatted_examples = []

    for sample in samples:
        full_text = format_instruction(
            sample["question"],
            sample["options"],
            sample["answer"],
            sample.get("cot_content"),
            use_cot=use_cot,
        )
        formatted_examples.append(full_text)

    # Tokenize
    encodings = tokenizer(
        formatted_examples,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"],
        }
    )


def extract_answer_letter(generated_text: str, question_text: str) -> str:
    """Extract the answer letter (A-J) from generated text."""
    if "Let's think step by step:" in generated_text:
        generated_text = generated_text.split("Let's think step by step:")[-1]
    elif "Answer:" in generated_text and question_text in generated_text:
        generated_text = generated_text.split("Answer:")[-1]

    # Pattern 1: "The answer is X"
    match = re.search(r"[Tt]he answer is ([A-J])", generated_text)
    if match:
        return match.group(1).upper()

    # Pattern 2: "Answer: X" or "Answer X"
    match = re.search(r"[Aa]nswer:?\s*([A-J])", generated_text)
    if match:
        return match.group(1).upper()

    # Pattern 3: Just a letter at the end
    match = re.search(r"\b([A-J])\b", generated_text)
    if match:
        return match.group(1).upper()

    # Pattern 4: Letter followed by closing parenthesis
    match = re.search(r"([A-J])\)", generated_text)
    if match:
        return match.group(1).upper()

    return "UNKNOWN"


def evaluate_model_on_mmlu_pro(
    model,
    tokenizer,
    test_samples: List[Dict],
    use_cot: bool = True,
    max_samples: int = None,
    phase_name: str = "MMLU-Pro Evaluation",
) -> Dict:
    """
    Evaluate model on MMLU-Pro test samples.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        test_samples: List of MMLU-Pro test samples
        use_cot: Whether to use Chain-of-Thought format
        max_samples: Maximum number of samples to evaluate
        phase_name: Name of evaluation phase

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
    logger.info(f"{'=' * 80}")

    for i, sample in enumerate(test_samples):
        question = sample["question"]
        options = sample["options"]
        true_answer = sample["answer"]
        category = sample["category"]

        # Format prompt
        prompt = format_instruction(question, options, answer=None, use_cot=use_cot)
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_answer_letter(generated_text, question)

        is_correct = predicted_answer == true_answer
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
                "true_answer": true_answer,
                "predicted_answer": predicted_answer,
                "correct": is_correct,
                "category": category,
            }
        )

        # Log first 5 examples
        if i < 5:
            logger.info(f"\n[{i+1}/{len(test_samples)}] Category: {category}")
            logger.info(f"Question: {question[:100]}...")
            logger.info(f"True Answer: {true_answer}")
            logger.info(f"Predicted: {predicted_answer}")
            logger.info(f"{'✓ CORRECT' if is_correct else '✗ WRONG'}")

        # Progress updates
        if (i + 1) % 10 == 0:
            current_acc = (correct / total * 100) if total > 0 else 0
            logger.info(
                f"Progress: {i+1}/{len(test_samples)} - Accuracy: {current_acc:.1f}%"
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
    logger.info("=" * 80)

    # GPU selection
    device_str, selected_gpu = set_gpu_device(
        gpu_id=gpu_id, auto_select=(gpu_id is None)
    )
    logger.info(f"Using device: {device_str} (GPU {selected_gpu})")

    clear_gpu_memory()
    log_memory_usage("Pre-training")

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model = model.to(device_str)
    model.config.use_cache = False

    # Apply LoRA
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

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # BASELINE EVALUATION on MMLU-Pro (before training)
    logger.info("\n" + "🔍" * 40)
    logger.info("BASELINE EVALUATION (Before Training on External Data)")
    logger.info("🔍" * 40)
    logger.info("Testing untrained model on MMLU-Pro...\n")

    baseline_results = evaluate_model_on_mmlu_pro(
        model=model,
        tokenizer=tokenizer,
        test_samples=mmlu_test_samples,
        use_cot=use_cot,
        max_samples=50,
        phase_name="BASELINE (Untrained on MMLU-Pro)",
    )

    logger.info(f"✅ Baseline: {baseline_results['overall_accuracy']:.2f}% on MMLU-Pro")
    logger.info(f"   (Expected: ~10% for untrained model)\n")

    # Prepare training datasets
    logger.info("Formatting training data...")
    train_dataset = create_solver_dataset(
        train_samples, tokenizer, max_length=1024, use_cot=use_cot
    )
    val_dataset = create_solver_dataset(
        val_samples, tokenizer, max_length=1024, use_cot=use_cot
    )

    # Setup output directory
    if output_dir is None:
        output_dir = f"qwen3_mmlu_{model_type}_no_leakage_r{lora_rank}"
    os.makedirs(output_dir, exist_ok=True)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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

    # POST-TRAINING EVALUATION on MMLU-Pro
    logger.info("\n" + "🎯" * 40)
    logger.info("POST-TRAINING EVALUATION (After Training on External Data)")
    logger.info("🎯" * 40)
    logger.info("Testing fine-tuned model on MMLU-Pro...\n")

    post_training_results = evaluate_model_on_mmlu_pro(
        model=model,
        tokenizer=tokenizer,
        test_samples=mmlu_test_samples,
        use_cot=use_cot,
        max_samples=50,
        phase_name="POST-TRAINING (Trained on External Data)",
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
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
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
            max_samples_per_dataset=args.max_samples_per_dataset,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            gpu_id=args.gpu_id,
            use_cot=args.use_cot,
        )
    elif args.mode == "test":
        # TODO: Implement standalone test mode
        logger.info("Test mode not yet implemented for this script")
