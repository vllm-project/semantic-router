"""
MMLU-Pro Category Classification with Qwen3 Generative Fine-tuning + LoRA
Fine-tunes Qwen3-0.6B as an instruction-following model to GENERATE category labels.

**CORRECT APPROACH**: Uses Qwen3 as a generative model (text-to-text)
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
import os
import sys

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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

from common_lora_utils import (  # noqa: E402
    clear_gpu_memory,
    log_memory_usage,
    select_training_split,
    set_gpu_device,
    setup_logging,
)

# Setup logging
logger = setup_logging()

# Standard category set for classification tasks (14 categories)
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


def get_qwen3_target_modules() -> list[str]:
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


class MMLU_Dataset:  # noqa: N801
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

            training_split_name, training_split = select_training_split(
                dataset, self.dataset_name
            )
            all_texts = training_split["question"]
            all_labels = training_split["category"]

            logger.info(
                f"Total samples in {training_split_name} split: {len(all_texts)}"
            )

            # Group samples by category
            category_samples = {}
            for text, label in zip(all_texts, all_labels, strict=False):
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
        unique_labels = sorted(set(labels))
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

        logger.info("Dataset sizes:")
        logger.info(f"  Train: {len(train_texts)}")
        logger.info(f"  Validation: {len(val_texts)}")
        logger.info(f"  Test: {len(test_texts)}")

        return {
            "train": (train_texts, train_labels),
            "validation": (val_texts, val_labels),
            "test": (test_texts, test_labels),
        }


def format_instruction(
    question: str, category: str | None = None
) -> list[dict[str, str]]:
    """
    Format a question-category pair as chat messages for proper instruction fine-tuning.

    Uses Qwen3's ChatML format with special tokens to separate user input from assistant output.
    This ensures the model only trains on generating the category name (1-2 tokens), not the
    entire instruction (~200+ tokens), resulting in 100x more efficient training!

    Args:
        question: The question text
        category: The category label (None for inference)

    Returns:
        List of message dicts with 'role' and 'content' keys
        Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    instruction = INSTRUCTION_TEMPLATE.format(question=question)

    # User message (the instruction/question)
    messages = [{"role": "user", "content": instruction}]

    if category is not None:
        # Assistant message (the category name)
        # This is just 1-2 tokens - much more efficient than training on entire sequence!
        messages.append({"role": "assistant", "content": category})

    return messages


def create_generative_dataset(
    texts: list[str], labels: list[str], tokenizer, max_length=512
):
    """
    Create dataset in chat format for proper instruction fine-tuning.

    Uses tokenizer.apply_chat_template() to format messages with special tokens.
    This ensures:
    - User input (instruction) and assistant output (category) are properly separated
    - Model trains ONLY on the category name (1-2 tokens), not the instruction (200+ tokens)
    - Training efficiency: Focuses 100% of gradient updates on classification tokens
    - Inference format matches training format exactly
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for text, label in zip(texts, labels, strict=False):
        # Get messages (user instruction + assistant category)
        messages = format_instruction(text, label)

        # Apply chat template to add special tokens
        # add_generation_prompt=False because we already have the assistant response
        # Disable thinking mode to train model for direct classification
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        # Tokenize the full conversation
        full_encoding = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # Now tokenize just the user part (without assistant response) to find where to mask
        user_messages = [messages[0]]  # Only user message
        user_text = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        user_encoding = tokenizer(user_text, add_special_tokens=False)

        # Create labels: -100 for instruction, actual tokens for assistant response
        labels = full_encoding["input_ids"].copy()
        user_length = len(user_encoding["input_ids"])

        # Mask the instruction part (user message + prompt)
        for i in range(min(user_length, len(labels))):
            labels[i] = -100

        # Mask padding tokens
        for i, token_id in enumerate(full_encoding["input_ids"]):
            if token_id == tokenizer.pad_token_id:
                labels[i] = -100

        input_ids_list.append(full_encoding["input_ids"])
        labels_list.append(labels)
        attention_mask_list.append(full_encoding["attention_mask"])

    return Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }
    )


def compute_metrics_generative(eval_pred, tokenizer, label2id):
    """
    Compute metrics for generative classification during training.

    Since we can't do actual generation during training (too slow),
    we compute a proxy metric: token-level accuracy at the answer position.

    This checks if the model predicts the correct category token.
    """
    import numpy as np  # noqa: PLC0415

    predictions, labels = eval_pred

    # predictions shape: (batch_size, seq_len, vocab_size) or (batch_size, seq_len)
    # labels shape: (batch_size, seq_len)

    # Ensure predictions is a numpy array
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    # Get predicted tokens (argmax over vocabulary if logits, otherwise use as-is)
    if len(predictions.shape) == 3:  # noqa: PLR2004
        # Logits shape: apply argmax to get token IDs
        pred_tokens = np.argmax(predictions, axis=-1)
    elif len(predictions.shape) == 2:  # noqa: PLR2004
        # Already token IDs
        pred_tokens = predictions
    else:
        # Unexpected shape, flatten or return zero metrics
        logger.warning(
            f"Unexpected predictions shape: {predictions.shape}. Returning zero metrics."
        )
        return {"token_accuracy": 0.0}

    # Only evaluate non-padding positions (labels != -100)
    mask = labels != -100  # noqa: PLR2004

    # Token-level accuracy
    correct_tokens = (pred_tokens == labels) & mask
    token_accuracy = correct_tokens.sum() / mask.sum() if mask.sum() > 0 else 0.0

    # Calculate perplexity from loss
    # Note: This is an approximation since we don't have access to loss here

    return {
        "token_accuracy": float(token_accuracy),
    }


def _parse_generated_category(generated_text: str) -> str:
    generated_text = (
        generated_text.replace("<think>", "").replace("</think>", "").strip()
    )
    answer_text = generated_text.split("\n")[0].strip().strip(".,!?;:").lower()

    predicted_category = "unknown"
    for category in REQUIRED_CATEGORIES:
        if answer_text.startswith(category.lower()):
            predicted_category = category.lower()
            break

    if predicted_category == "unknown" and answer_text:
        words = answer_text.split()
        if len(words) >= 2:  # noqa: PLR2004
            predicted_category = " ".join(words[:2])
        elif len(words) == 1:
            predicted_category = words[0]
        else:
            predicted_category = answer_text

    return predicted_category


def _load_and_configure_model(model_name, lora_rank, lora_alpha, lora_dropout, device_str):
    logger.info(f"Loading Qwen3 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Using F32 dtype for training (more stable than BF16)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    device = torch.device(device_str)
    model = model.to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

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

    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable: {name}")
            break

    return model, tokenizer, device


def _create_training_args(output_dir, num_epochs, batch_size, learning_rate, num_workers):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=max(
            1, 16 // batch_size
        ),
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        save_total_limit=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=False,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        prediction_loss_only=True,
        dataloader_pin_memory=False,
        auto_find_batch_size=False,
    )


def _setup_gpu(gpu_id):
    device_str, selected_gpu = set_gpu_device(
        gpu_id=gpu_id, auto_select=(gpu_id is None)
    )
    logger.info(f"Using device: {device_str} (GPU {selected_gpu})")

    # CRITICAL: Set CUDA_VISIBLE_DEVICES early to prevent DataParallel from using other GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    # After setting CUDA_VISIBLE_DEVICES, GPU will be referred to as cuda:0
    device_str = "cuda:0"
    logger.info(f"Set CUDA_VISIBLE_DEVICES={selected_gpu}, using device: {device_str}")

    clear_gpu_memory()
    log_memory_usage("Pre-training")
    return device_str


def _save_model_and_mapping(trainer, tokenizer, output_dir, dataset_loader):
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping = {
        "label2id": dataset_loader.label2id,
        "id2label": dataset_loader.id2label,
        "instruction_template": INSTRUCTION_TEMPLATE,
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)

    logger.info(f"Model saved to: {output_dir}")


def _evaluate_generative_accuracy(model, tokenizer, val_texts, val_labels, num_test_samples):
    model.eval()

    correct = 0
    total = 0

    logger.info(f"Testing on {num_test_samples} validation samples...")

    for i in range(num_test_samples):
        question = val_texts[i]
        true_category = val_labels[i]

        messages = format_instruction(question, category=None)

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>"),
                ],
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        predicted_category = _parse_generated_category(generated_text)

        is_correct = predicted_category == true_category.lower()
        if is_correct:
            correct += 1
        total += 1

        if i < 5 or i >= num_test_samples - 5:  # noqa: PLR2004
            logger.info(f"\n[{i+1}/{num_test_samples}] Question: {question[:100]}...")
            logger.info(f"  True: {true_category}")
            logger.info(f"  Predicted: {predicted_category}")
            logger.info(f"  {'CORRECT' if is_correct else '✗ WRONG'}")

    accuracy = (correct / total * 100) if total > 0 else 0
    return correct, total, accuracy


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
    output_dir: str | None = None,
    gpu_id: int | None = None,
):
    """Main training function for generative Qwen3 classification.

    Args:
        max_samples_per_category: Maximum samples per category (default: 150).
                                 With 14 categories, this gives ~2100 total samples.
    """
    logger.info("Starting Qwen3 Generative Classification Fine-tuning")
    logger.info("Training Qwen3 to GENERATE category labels (instruction-following)")

    device_str = _setup_gpu(gpu_id)

    # Load dataset
    dataset_loader = MMLU_Dataset()
    datasets = dataset_loader.prepare_datasets(max_samples_per_category)

    train_texts, train_labels = datasets["train"]
    val_texts, val_labels = datasets["validation"]

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    logger.info(f"Categories: {len(dataset_loader.label2id)}")

    # Load tokenizer and model
    model, tokenizer, device = _load_and_configure_model(
        model_name, lora_rank, lora_alpha, lora_dropout, device_str
    )

    # Prepare datasets in generative format
    logger.info("Formatting dataset for instruction-following...")
    train_dataset = create_generative_dataset(train_texts, train_labels, tokenizer)
    val_dataset = create_generative_dataset(val_texts, val_labels, tokenizer)

    logger.info("Example training input:")
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

    training_args = _create_training_args(output_dir, num_epochs, batch_size, learning_rate, num_workers)

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

    _save_model_and_mapping(trainer, tokenizer, output_dir, dataset_loader)

    # Test generation on MMLU-Pro validation data
    logger.info("\n" + "=" * 50)
    logger.info("Testing generation on MMLU-Pro validation data:")
    logger.info("=" * 50)

    num_test_samples = min(200, len(val_texts))
    correct, total, accuracy = _evaluate_generative_accuracy(
        model, tokenizer, val_texts, val_labels, num_test_samples
    )

    logger.info("\n" + "=" * 50)
    logger.info(f"Validation Accuracy: {correct}/{total} = {accuracy:.2f}%")
    logger.info("=" * 50)

    log_memory_usage("Post-training")


def demo_inference(model_path: str, model_name: str = "Qwen/Qwen3-0.6B"):
    """Demonstrate inference with trained generative model."""
    logger.info(f"Loading generative Qwen3 model from: {model_path}")

    try:
        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.json")) as f:
            json.load(f)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model with F32 (matches training)
        logger.info("Using F32 dtype for inference (matches training)")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
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

        for example in test_examples:
            # Format using chat template
            messages = format_instruction(example, category=None)

            # Apply chat template with generation prompt
            # Disable thinking mode for direct classification output
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    ],
                )

            # Decode only the generated part (skip the input prompt)
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            category = _parse_generated_category(generated_text)

            print(f"\nQuestion: {example}")
            print(f"Generated: {generated_text[:50]}...")
            print(f"Predicted Category: {category}")
            print("-" * 80)

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback  # noqa: PLC0415

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
