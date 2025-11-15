"""
Example: RL-enabled Intent Classification Training

Shows how to integrate PPO fine-tuning on top of supervised LoRA training.
This is a minimal example to demonstrate the two-phase training pipeline.

Usage:
    # Supervised LoRA only (default)
    python train_with_rl_example.py --mode supervised

    # Supervised + RL (PPO)
    python train_with_rl_example.py --mode supervised_then_rl

    # RL only (fine-tune existing model)
    python train_with_rl_example.py --mode rl_only --pretrained-model path/to/model
"""

import argparse
import logging
import os
from typing import Optional

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_dummy_loader(batch_size: int = 8, num_samples: int = 100):
    """Create dummy data loader for demo purposes."""

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
            self.texts = [
                f"Sample text {i} for intent classification" for i in range(num_samples)
            ]
            self.labels = torch.randint(0, 14, (num_samples,))  # 14 intent categories

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    dataset = DummyDataset(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_supervised_lora(
    model_name: str = "bert-base-uncased",
    num_epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    output_dir: str = "models/intent_classifier_supervised",
):
    """
    Train model with supervised LoRA (existing pipeline).

    This is a wrapper around ft_linear_lora.py functionality.
    """
    import sys

    sys.path.insert(
        0,
        os.path.join(os.path.dirname(__file__), "classifier_model_fine_tuning_lora"),
    )

    from ft_linear_lora import main as train_lora

    logger.info("=" * 80)
    logger.info("PHASE 1: Supervised LoRA Training")
    logger.info("=" * 80)

    train_lora(
        model_name=model_name,
        lora_rank=8,
        lora_alpha=16,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
        max_samples=500,  # Small for demo
    )

    logger.info(f"Supervised model saved to: {output_dir}")
    return output_dir


def train_with_rl(
    pretrained_model_path: str,
    model_name: str = "bert-base-uncased",
    num_episodes: int = 3,
    batch_size: int = 8,
    reward_metric: str = "accuracy",
    output_dir: str = "models/intent_classifier_rl",
):
    """
    Fine-tune supervised model with PPO RL.

    Args:
        pretrained_model_path: Path to supervised LoRA model
        model_name: Base model name
        num_episodes: Number of PPO episodes
        batch_size: Batch size
        reward_metric: Reward metric ("accuracy", "f1", "calibration")
        output_dir: Output directory
    """
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    from rl_ppo_trainer import train_with_rl as run_ppo_training

    logger.info("=" * 80)
    logger.info("PHASE 2: RL Fine-tuning with PPO")
    logger.info("=" * 80)

    # Load pretrained supervised model
    logger.info(f"Loading pretrained model from: {pretrained_model_path}")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=14, torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base_model, pretrained_model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dummy dataloaders
    train_loader = create_dummy_loader(batch_size=batch_size, num_samples=200)
    val_loader = create_dummy_loader(batch_size=batch_size, num_samples=50)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    # Run PPO training
    ppo_config = {
        "learning_rate": 1e-5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "eps_clip": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "batch_size": batch_size,
        "update_epochs": 4,
    }

    metrics = run_ppo_training(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_episodes=num_episodes,
        reward_metric=reward_metric,
        device=device,
        **ppo_config,
    )

    logger.info("PPO training complete!")
    logger.info(f"Best episodic return: {metrics['best_episodic_return']:.4f}")

    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"RL-trained model saved to: {output_dir}")

    return output_dir, metrics


def main():
    parser = argparse.ArgumentParser(
        description="RL-enabled Intent Classification Training"
    )
    parser.add_argument(
        "--mode",
        choices=["supervised", "supervised_then_rl", "rl_only"],
        default="supervised_then_rl",
        help="Training mode",
    )
    parser.add_argument(
        "--model", default="bert-base-uncased", help="Base model name"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Supervised epochs")
    parser.add_argument(
        "--rl-episodes", type=int, default=3, help="RL episodes (PPO)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--reward-metric",
        choices=["accuracy", "f1", "calibration"],
        default="accuracy",
        help="RL reward metric",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Path to pretrained model (for rl_only mode)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Intent Classification Training with Optional RL")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.mode != "supervised":
        logger.info(f"RL reward metric: {args.reward_metric}")

    # Phase 1: Supervised LoRA
    if args.mode in ["supervised", "supervised_then_rl"]:
        supervised_dir = os.path.join(args.output_dir, "intent_classifier_supervised")

        supervised_model_path = train_supervised_lora(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=supervised_dir,
        )

        if args.mode == "supervised":
            logger.info("Supervised training complete!")
            return supervised_model_path

        pretrained_model_path = supervised_model_path

    elif args.mode == "rl_only":
        if args.pretrained_model is None:
            raise ValueError(
                "--pretrained-model required for rl_only mode"
            )
        pretrained_model_path = args.pretrained_model

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Phase 2: RL Fine-tuning
    if args.mode in ["supervised_then_rl", "rl_only"]:
        rl_dir = os.path.join(args.output_dir, "intent_classifier_rl")

        rl_model_path, rl_metrics = train_with_rl(
            pretrained_model_path=pretrained_model_path,
            model_name=args.model,
            num_episodes=args.rl_episodes,
            batch_size=args.batch_size,
            reward_metric=args.reward_metric,
            output_dir=rl_dir,
        )

        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Final model saved to: {rl_model_path}")
        logger.info(f"Best episodic return: {rl_metrics['best_episodic_return']:.4f}")

        return rl_model_path

    return pretrained_model_path


if __name__ == "__main__":
    main()
