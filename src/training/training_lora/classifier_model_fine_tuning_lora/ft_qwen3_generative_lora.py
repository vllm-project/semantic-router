"""
MMLU-Pro Category Classification with Qwen3 Generative Fine-tuning + LoRA
Fine-tunes Qwen3-0.6B as an instruction-following model to GENERATE category labels.

✅ **CORRECT APPROACH**: Uses Qwen3 as a generative model (text-to-text)
   - Qwen3 generates category names as text
   - Standard causal language modeling (how Qwen3 was pre-trained)
   - Instruction-tuning format (like ChatGPT/Claude)
   - Expected accuracy: 70-85% (much better than classification head approach!)

🎯 **How it works**:
   Input:  "Classify this question: What is corporate law? Category:"
   Output: "law"

   The model learns to generate the category name as text, which is natural for a
   causal language model!

Usage:
    # Train with recommended parameters (150 samples per category = ~2100 total)
    python ft_qwen3_generative_lora.py --mode train --epochs 8 --lora-rank 16 --max-samples-per-category 150

    # Test with specific GPU
    python ft_qwen3_generative_lora.py --mode train --epochs 8 --gpu-id 2

    # Adjust batch size based on GPU memory (default: 4)
    python ft_qwen3_generative_lora.py --mode train --batch-size 8 --epochs 5

    # Quick test (10 samples per category = ~140 total)
    python ft_qwen3_generative_lora.py --mode train --epochs 1 --max-samples-per-category 10

    # Inference
    python ft_qwen3_generative_lora.py --mode test --model-path qwen3_generative_classifier

Model:
    - Qwen/Qwen3-0.6B (752M params, 28 layers, 32k context)
    - Fine-tuned with LoRA on instruction-following format
    - Generates category labels as text (natural for decoder models!)

Dataset:
    - TIGER-Lab/MMLU-Pro: 14 category academic question classification
    - Formatted as instruction-following pairs
    - Categories: biology, business, chemistry, computer science, economics,
                  engineering, health, history, law, math, other, philosophy,
                  physics, psychology
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
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

# Import common LoRA utilities
# Note: Using sys.path for standalone script compatibility.
# For package installations, use: from semantic_router.training.common_lora_utils import ...
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from common_lora_utils import (
    clear_gpu_memory,
    log_memory_usage,
    set_gpu_device,
    setup_logging,
)

# Setup logging
logger = setup_logging()

# Required categories to match legacy model (14 categories)
REQUIRED_CATEGORIES = [
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

# Instruction template for classification (improved with examples)
INSTRUCTION_TEMPLATE = """You are an expert academic classifier. Classify the following question into exactly ONE category. Respond with ONLY the category name.

Categories: biology, business, chemistry, computer science, economics, engineering, health, history, law, math, other, philosophy, physics, psychology

Examples:
Q: What is the optimal capital structure for a corporation?
A: business

Q: How do neurons transmit signals?
A: biology

Q: What are the principles of contract law?
A: law

Now classify this question:
Q: {question}
A:"""


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


class MMLU_Dataset:
    """Dataset class for MMLU-Pro category classification."""

    def __init__(self, dataset_name="TIGER-Lab/MMLU-Pro"):
        self.dataset_name = dataset_name
        self.label2id = {}
        self.id2label = {}

    def load_huggingface_dataset(self, max_samples_per_category=150):
        """Load the MMLU-Pro dataset from HuggingFace with balanced sampling.

        Args:
            max_samples_per_category: Maximum number of samples to take from each category.
                                     Default: 150 per category (14 categories = ~2100 total)
        """
        logger.info(f"Loading dataset from HuggingFace: {self.dataset_name}")

        try:
            dataset = load_dataset(self.dataset_name)
            logger.info(f"Dataset splits: {dataset.keys()}")

            all_texts = dataset["test"]["question"]
            all_labels = dataset["test"]["category"]

            logger.info(f"Total samples in dataset: {len(all_texts)}")

            # Group samples by category
            category_samples = {}
            for text, label in zip(all_texts, all_labels):
                if label not in category_samples:
                    category_samples[label] = []
                category_samples[label].append(text)

            logger.info(f"Available categories: {sorted(category_samples.keys())}")

            # Use samples per category directly
            available_required_categories = [
                cat for cat in REQUIRED_CATEGORIES if cat in category_samples
            ]

            target_samples_per_category = max_samples_per_category

            # Collect balanced samples
            filtered_texts = []
            filtered_labels = []
            category_counts = {}

            for category in available_required_categories:
                if category in category_samples:
                    samples_to_take = min(
                        target_samples_per_category, len(category_samples[category])
                    )
                    category_texts = category_samples[category][:samples_to_take]
                    filtered_texts.extend(category_texts)
                    filtered_labels.extend([category] * len(category_texts))
                    category_counts[category] = len(category_texts)

            logger.info(f"Final category distribution: {category_counts}")
            logger.info(f"Total filtered samples: {len(filtered_texts)}")

            return filtered_texts, filtered_labels

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def prepare_datasets(self, max_samples_per_category=150):
        """Prepare train/validation/test datasets.

        Args:
            max_samples_per_category: Maximum samples per category (default: 150)
        """
        texts, labels = self.load_huggingface_dataset(max_samples_per_category)

        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        ordered_labels = [cat for cat in REQUIRED_CATEGORIES if cat in unique_labels]

        self.label2id = {label: idx for idx, label in enumerate(ordered_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        logger.info(f"Found {len(ordered_labels)} categories: {ordered_labels}")

        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.4, random_state=42, stratify=labels
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels,
        )

        logger.info(f"Dataset sizes:")
        logger.info(f"  Train: {len(train_texts)}")
        logger.info(f"  Validation: {len(val_texts)}")
        logger.info(f"  Test: {len(test_texts)}")

        return {
            "train": (train_texts, train_labels),
            "validation": (val_texts, val_labels),
            "test": (test_texts, test_labels),
        }


def format_instruction(question: str, category: str = None) -> str:
    """
    Format a question-category pair as an instruction-following example.

    Args:
        question: The question text
        category: The category label (None for inference)

    Returns:
        Formatted instruction string (with or without answer)
    """
    instruction = INSTRUCTION_TEMPLATE.format(question=question)

    if category is not None:
        # Training format: instruction + answer
        return f"{instruction} {category}"
    else:
        # Inference format: instruction only
        return instruction


def create_generative_dataset(
    texts: List[str], labels: List[str], tokenizer, max_length=512
):
    """
    Create dataset in generative format for instruction-following.

    Format: "Question: ... Category: {label}"
    The model learns to generate the category name.
    """
    formatted_examples = []

    for text, label in zip(texts, labels):
        # Create full text: instruction + answer
        full_text = format_instruction(text, label)
        formatted_examples.append(full_text)

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
            "labels": encodings["input_ids"],  # Standard causal LM format
        }
    )


def compute_metrics_generative(eval_pred, tokenizer, label2id):
    """
    Compute metrics for generative classification during training.

    Since we can't do actual generation during training (too slow),
    we compute a proxy metric: token-level accuracy at the answer position.

    This checks if the model predicts the correct category token.
    """
    import numpy as np

    predictions, labels = eval_pred

    # predictions shape: (batch_size, seq_len, vocab_size) or (batch_size, seq_len)
    # labels shape: (batch_size, seq_len)

    # Ensure predictions is a numpy array
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    # Get predicted tokens (argmax over vocabulary if logits, otherwise use as-is)
    if len(predictions.shape) == 3:
        # Logits shape: apply argmax to get token IDs
        pred_tokens = np.argmax(predictions, axis=-1)
    elif len(predictions.shape) == 2:
        # Already token IDs
        pred_tokens = predictions
    else:
        # Unexpected shape, flatten or return zero metrics
        logger.warning(
            f"Unexpected predictions shape: {predictions.shape}. Returning zero metrics."
        )
        return {"token_accuracy": 0.0}

    # Only evaluate non-padding positions (labels != -100)
    mask = labels != -100

    # Token-level accuracy
    correct_tokens = (pred_tokens == labels) & mask
    token_accuracy = correct_tokens.sum() / mask.sum() if mask.sum() > 0 else 0.0

    # Calculate perplexity from loss
    # Note: This is an approximation since we don't have access to loss here

    return {
        "token_accuracy": float(token_accuracy),
    }


def main(
    model_name: str = "Qwen/Qwen3-0.6B",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,  # Lower dropout for small model
    num_epochs: int = 8,  # More epochs for 0.6B
    batch_size: int = 4,  # Configurable batch size (adjust based on GPU memory)
    learning_rate: float = 3e-4,  # Higher LR for small model
    max_samples_per_category: int = 150,  # Samples per category for balanced dataset
    num_workers: int = 0,  # Number of dataloader workers (0=single process, 2-4 for multiprocessing)
    output_dir: str = None,
    gpu_id: Optional[int] = None,
):
    """Main training function for generative Qwen3 classification.

    Args:
        max_samples_per_category: Maximum samples per category (default: 150).
                                 With 14 categories, this gives ~2100 total samples.
    """
    logger.info("Starting Qwen3 Generative Classification Fine-tuning")
    logger.info("Training Qwen3 to GENERATE category labels (instruction-following)")

    # GPU selection using utility function
    device_str, selected_gpu = set_gpu_device(
        gpu_id=gpu_id, auto_select=(gpu_id is None)
    )
    logger.info(f"Using device: {device_str} (GPU {selected_gpu})")

    clear_gpu_memory()
    log_memory_usage("Pre-training")

    # Load dataset
    dataset_loader = MMLU_Dataset()
    datasets = dataset_loader.prepare_datasets(max_samples_per_category)

    train_texts, train_labels = datasets["train"]
    val_texts, val_labels = datasets["validation"]

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    logger.info(f"Categories: {len(dataset_loader.label2id)}")

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

    # Move to GPU using device from set_gpu_device utility
    model = model.to(device_str)

    # Prepare model for training
    model.config.use_cache = False  # Required for training

    # Create LoRA configuration
    target_modules = get_qwen3_target_modules()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Correct task type for generation
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

    # Ensure model is in training mode and enable gradients
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable: {name}")
            break  # Just log first one to verify

    # Prepare datasets in generative format
    logger.info("Formatting dataset for instruction-following...")
    train_dataset = create_generative_dataset(train_texts, train_labels, tokenizer)
    val_dataset = create_generative_dataset(val_texts, val_labels, tokenizer)

    logger.info(f"Example training input:")
    logger.info(tokenizer.decode(train_dataset[0]["input_ids"][:100]))

    # Setup output directory
    if output_dir is None:
        output_dir = f"qwen3_generative_classifier_r{lora_rank}"
    os.makedirs(output_dir, exist_ok=True)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments (optimized for memory and stability)
    # Note: batch_size is configurable via function parameter
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,  # Configurable via parameter
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(
            1, 16 // batch_size
        ),  # Maintain effective batch size of 16, minimum 1
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",  # Don't save intermediate checkpoints (saves disk space!)
        save_total_limit=1,  # Keep only 1 checkpoint
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=False,  # Disable fp16 to avoid gradient issues
        gradient_checkpointing=False,  # Disable to avoid gradient issues
        dataloader_num_workers=num_workers,  # Configurable workers (0=single process, 2-4=multiprocessing)
        remove_unused_columns=False,  # Keep all columns
        max_grad_norm=1.0,  # Gradient clipping for stability
        optim="adamw_torch",  # Use PyTorch AdamW
        prediction_loss_only=True,  # Only compute loss, don't collect predictions (saves memory!)
    )

    # Create trainer (no compute_metrics needed since prediction_loss_only=True)
    # Real accuracy will be computed at the end using actual generation
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

    # Save label mapping
    label_mapping = {
        "label2id": dataset_loader.label2id,
        "id2label": dataset_loader.id2label,
        "instruction_template": INSTRUCTION_TEMPLATE,
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)

    logger.info(f"Model saved to: {output_dir}")

    # Test generation on MMLU-Pro validation data
    logger.info("\n" + "=" * 50)
    logger.info("Testing generation on MMLU-Pro validation data:")
    logger.info("=" * 50)

    model.eval()

    # Use validation data for testing
    num_test_samples = min(20, len(val_texts))  # Test on 20 samples
    correct = 0
    total = 0

    logger.info(f"Testing on {num_test_samples} validation samples...")

    for i in range(num_test_samples):
        question = val_texts[i]
        true_category = val_labels[i]

        prompt = format_instruction(question, category=None)
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,  # Greedy decoding for evaluation
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the category (text after "A:" or "Category:")
        if "A:" in generated_text:
            answer_text = generated_text.split("A:")[-1].strip()
        elif "Category:" in generated_text:
            answer_text = generated_text.split("Category:")[-1].strip()
        else:
            answer_text = ""

        # Clean up answer (take first line, remove punctuation at end)
        answer_text = answer_text.split("\n")[0].strip().strip(".,!?;:").lower()

        # Match against known categories (handle multi-word categories like "computer science")
        predicted_category = "unknown"
        for category in REQUIRED_CATEGORIES:
            if answer_text.startswith(category.lower()):
                predicted_category = category.lower()
                break

        # If no match, take first 2 words (for "computer science" etc)
        if predicted_category == "unknown" and answer_text:
            words = answer_text.split()
            if len(words) >= 2:
                predicted_category = " ".join(words[:2])
            elif len(words) == 1:
                predicted_category = words[0]
            else:
                predicted_category = answer_text

        is_correct = predicted_category == true_category.lower()
        if is_correct:
            correct += 1
        total += 1

        # Log first 5 and last 5 examples
        if i < 5 or i >= num_test_samples - 5:
            logger.info(f"\n[{i+1}/{num_test_samples}] Question: {question[:100]}...")
            logger.info(f"  True: {true_category}")
            logger.info(f"  Predicted: {predicted_category}")
            logger.info(f"  {'✓ CORRECT' if is_correct else '✗ WRONG'}")

    accuracy = (correct / total * 100) if total > 0 else 0
    logger.info("\n" + "=" * 50)
    logger.info(f"Validation Accuracy: {correct}/{total} = {accuracy:.2f}%")
    logger.info("=" * 50)

    log_memory_usage("Post-training")


def demo_inference(model_path: str, model_name: str = "Qwen/Qwen3-0.6B"):
    """Demonstrate inference with trained generative model."""
    logger.info(f"Loading generative Qwen3 model from: {model_path}")

    try:
        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            mapping_data = json.load(f)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model with appropriate dtype
        # Check for GPU capability and use float16 only if supported
        use_fp16 = False
        if torch.cuda.is_available():
            # Check if GPU supports efficient float16 (compute capability >= 7.0)
            try:
                compute_capability = torch.cuda.get_device_capability()
                use_fp16 = (
                    compute_capability[0] >= 7
                )  # Volta and newer support efficient FP16
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

        # Test examples
        test_examples = [
            "What is the best strategy for corporate mergers and acquisitions?",
            "How do antitrust laws affect business competition?",
            "What are the psychological factors that influence consumer behavior?",
            "Explain the legal requirements for contract formation",
            "What is the difference between civil and criminal law?",
            "How does cognitive bias affect decision making?",
            "What are the key principles of quantum mechanics?",
            "Explain the process of cellular respiration in biology",
        ]

        logger.info("Running inference...")
        correct = 0
        total = 0

        for example in test_examples:
            prompt = format_instruction(example, category=None)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract category (handle both "A:" and "Category:" formats)
            if "A:" in generated_text:
                answer_text = generated_text.split("A:")[-1].strip()
            elif "Category:" in generated_text:
                answer_text = generated_text.split("Category:")[-1].strip()
            else:
                answer_text = ""

            # Clean up and match against known categories
            answer_text = answer_text.split("\n")[0].strip().strip(".,!?;:").lower()

            category = "unknown"
            for cat in REQUIRED_CATEGORIES:
                if answer_text.startswith(cat.lower()):
                    category = cat
                    break

            # If no match, take first 2 words
            if category == "unknown" and answer_text:
                words = answer_text.split()
                category = (
                    " ".join(words[:2])
                    if len(words) >= 2
                    else words[0] if words else "unknown"
                )

            print(f"\nQuestion: {example}")
            print(f"Generated: {generated_text[len(prompt):50]}...")
            print(f"Predicted Category: {category}")
            print("-" * 80)

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3 Generative Classification (Instruction-Following)"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Qwen3 model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (adjust based on GPU memory: 1-2 for small GPUs, 4-8 for medium, 8-16 for large). Gradient accumulation auto-adjusts to maintain effective batch size of 16.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=150,
        help="Maximum samples per category for balanced training (default: 150 per category = ~2100 total with 14 categories)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers (0=single process for debugging, 2-4=multiprocessing for better performance)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        default="qwen3_generative_classifier_r16",
        help="Path to saved model for inference",
    )

    args = parser.parse_args()

    # GPU device selection is handled in main() and demo_inference() functions
    # using the set_gpu_device() utility function for consistency

    if args.mode == "train":
        main(
            model_name=args.model,
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
        )
    elif args.mode == "test":
        demo_inference(args.model_path, args.model)
