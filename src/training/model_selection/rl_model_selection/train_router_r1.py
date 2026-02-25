#!/usr/bin/env python3
"""
Router-R1 Training Script

Implements RL-based router training following the Router-R1 paper (arXiv:2506.09033).
The router learns to make multi-round routing decisions by:
1. Generating "Think" actions for reasoning
2. Generating "Route" actions to select models
3. Receiving rewards based on format, outcome, and cost

Usage:
    python train_router_r1.py --config configs/router_r1_config.yaml

References:
    - Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning
    - https://arxiv.org/abs/2506.09033
"""

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from trl import PPOConfig, PPOTrainer

    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    logging.warning("TRL not installed. Using custom RL training loop.")

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RouterR1Config:
    """Configuration for Router-R1 training."""

    # Model configuration
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    model_max_length: int = 2048

    # Training configuration
    learning_rate: float = 1e-5
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # RL-specific configuration
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus

    # Reward configuration
    format_reward_weight: float = 0.1
    outcome_reward_weight: float = 0.7
    cost_reward_weight: float = 0.2

    # Model costs ($/1M tokens)
    model_costs: Dict[str, float] = field(
        default_factory=lambda: {
            "gpt-4": 30.0,
            "claude-3-sonnet": 15.0,
            "mistral-7b": 0.5,
            "llama3-8b": 0.25,
        }
    )

    # Output configuration
    output_dir: str = "./checkpoints/router_r1"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    # Data configuration
    train_data_path: str = "./data/train_routing.json"
    eval_data_path: str = "./data/eval_routing.json"

    # Hardware
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> "RouterR1Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


def custom_collate_fn(batch):
    """Custom collate function to handle the example field properly."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    examples = [item["example"] for item in batch]  # Keep as list of dicts
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "example": examples,
    }


class RoutingDataset(Dataset):
    """Dataset for router training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_data(data_path)

    def _load_data(self, path: str) -> List[Dict]:
        """Load training examples from JSON."""
        if not os.path.exists(path):
            logger.warning(f"Data file not found: {path}. Using synthetic data.")
            return self._generate_synthetic_data()

        with open(path, "r") as f:
            return json.load(f)

    def _generate_synthetic_data(self, n_examples: int = 1000) -> List[Dict]:
        """Generate synthetic training data for demonstration."""
        examples = []

        query_types = [
            ("What is 2+2?", "math", "simple"),
            ("Explain quantum entanglement in detail", "physics", "complex"),
            ("Write a Python function to sort a list", "coding", "medium"),
            ("What is the capital of France?", "factual", "simple"),
            ("Analyze the themes in Shakespeare's Hamlet", "literature", "complex"),
        ]

        models = ["gpt-4", "claude-3-sonnet", "mistral-7b", "llama3-8b"]

        for i in range(n_examples):
            query, domain, complexity = random.choice(query_types)
            query = f"{query} (variation {i})"

            # Simple heuristic for optimal model
            if complexity == "simple":
                optimal = random.choice(["mistral-7b", "llama3-8b"])
            elif complexity == "medium":
                optimal = random.choice(["claude-3-sonnet", "mistral-7b"])
            else:
                optimal = random.choice(["gpt-4", "claude-3-sonnet"])

            examples.append(
                {
                    "query": query,
                    "query_type": domain,
                    "complexity": complexity,
                    "candidate_models": models,
                    "optimal_model": optimal,
                    "ground_truth": f"Answer to: {query}",
                }
            )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]

        # Format prompt for router
        prompt = self._format_router_prompt(example)

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "example": example,
        }

    def _format_router_prompt(self, example: Dict) -> str:
        """Format the routing prompt following Router-R1 style."""
        models_str = ", ".join(example["candidate_models"])

        prompt = f"""You are an intelligent router that decides which LLM should handle a query.

Available models: {models_str}

Query: {example["query"]}
Query type: {example.get("query_type", "unknown")}

Think step by step about which model would be best, then route the query.

<think>
Let me analyze this query:
- Query complexity: {example.get("complexity", "unknown")}
- Domain: {example.get("query_type", "unknown")}
- Cost considerations: cheaper models preferred for simple queries

Based on this analysis, I should route to the most appropriate model.
</think>

<route>"""

        return prompt


class RouterRewardModel:
    """Computes rewards for router actions."""

    def __init__(self, config: RouterR1Config):
        self.config = config
        self.model_costs = config.model_costs

    def compute_reward(
        self,
        predicted_model: str,
        optimal_model: str,
        generated_response: str,
        ground_truth: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward as weighted sum of:
        1. Format reward: Did the router follow the correct format?
        2. Outcome reward: Did it select the right model?
        3. Cost reward: Did it optimize for cost?
        """
        # Format reward: Check if output follows <think>...</think><route>...</route>
        format_reward = self._compute_format_reward(generated_response)

        # Outcome reward: Did we select the optimal model?
        outcome_reward = self._compute_outcome_reward(predicted_model, optimal_model)

        # Cost reward: Reward cheaper models (normalized)
        cost_reward = self._compute_cost_reward(predicted_model)

        # Weighted sum
        total_reward = (
            self.config.format_reward_weight * format_reward
            + self.config.outcome_reward_weight * outcome_reward
            + self.config.cost_reward_weight * cost_reward
        )

        return total_reward, {
            "format_reward": format_reward,
            "outcome_reward": outcome_reward,
            "cost_reward": cost_reward,
            "total_reward": total_reward,
        }

    def _compute_format_reward(self, response: str) -> float:
        """Reward for following correct output format."""
        has_think = "<think>" in response and "</think>" in response
        has_route = "<route>" in response

        if has_think and has_route:
            return 1.0
        elif has_route:
            return 0.5
        else:
            return 0.0

    def _compute_outcome_reward(self, predicted: str, optimal: str) -> float:
        """Reward for selecting the optimal model."""
        if predicted == optimal:
            return 1.0

        # Partial reward for similar models
        tier_map = {
            "gpt-4": 3,
            "claude-3-sonnet": 2,
            "mistral-7b": 1,
            "llama3-8b": 1,
        }

        pred_tier = tier_map.get(predicted, 0)
        opt_tier = tier_map.get(optimal, 0)

        if pred_tier == opt_tier:
            return 0.7  # Same tier
        elif abs(pred_tier - opt_tier) == 1:
            return 0.3  # Adjacent tier
        else:
            return 0.0  # Wrong tier

    def _compute_cost_reward(self, model: str) -> float:
        """Reward for cost efficiency (normalized)."""
        if model not in self.model_costs:
            return 0.5

        cost = self.model_costs[model]
        max_cost = max(self.model_costs.values())
        min_cost = min(self.model_costs.values())

        if max_cost == min_cost:
            return 1.0

        # Higher reward for lower cost
        normalized = (max_cost - cost) / (max_cost - min_cost)
        return normalized


class RouterR1Trainer:
    """Trainer for Router-R1 model."""

    def __init__(self, config: RouterR1Config):
        self.config = config
        self.device = torch.device(config.device)

        # Load model and tokenizer
        logger.info(f"Loading model: {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Reward model
        self.reward_model = RouterRewardModel(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Metrics
        self.train_metrics = []

    def train(
        self,
        train_dataset: RoutingDataset,
        eval_dataset: Optional[RoutingDataset] = None,
    ):
        """Main training loop."""
        logger.info("Starting Router-R1 training...")

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

        num_training_steps = len(dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps,
        )

        global_step = 0
        best_eval_reward = float("-inf")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_rewards = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                # Generate router output
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                examples = batch["example"]

                # Generate routing decision - use greedy for stability
                with torch.no_grad():
                    try:
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=30,
                            do_sample=False,  # Greedy for stability
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    except RuntimeError as e:
                        logger.warning(f"Generation failed: {e}")
                        torch.cuda.empty_cache()
                        # Skip this batch
                        continue

                # Decode and extract model selection
                batch_rewards = []
                for i, (output, example) in enumerate(zip(outputs, examples)):
                    generated = self.tokenizer.decode(output, skip_special_tokens=True)
                    predicted_model = self._extract_model(
                        generated, example["candidate_models"]
                    )

                    reward, reward_details = self.reward_model.compute_reward(
                        predicted_model=predicted_model,
                        optimal_model=example["optimal_model"],
                        generated_response=generated,
                        ground_truth=example.get("ground_truth", ""),
                    )
                    batch_rewards.append(reward)

                # Compute policy gradient loss (simplified REINFORCE)
                avg_reward = np.mean(batch_rewards)
                epoch_rewards.extend(batch_rewards)

                # Forward pass for loss computation
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                # Scale loss by reward (REINFORCE) with stability clamp
                reward_scale = max(
                    0.1, min(0.9, 1.0 - avg_reward)
                )  # Clamp to [0.1, 0.9]
                loss = outputs.loss * reward_scale

                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Skipping batch {batch_idx} due to NaN/Inf loss")
                    continue

                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                # Logging
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_reward": f"{avg_reward:.4f}",
                    }
                )

                if global_step % self.config.logging_steps == 0:
                    self.train_metrics.append(
                        {
                            "step": global_step,
                            "loss": loss.item(),
                            "reward": avg_reward,
                        }
                    )

                # Evaluation
                if eval_dataset and global_step % self.config.eval_steps == 0:
                    eval_reward = self.evaluate(eval_dataset)
                    logger.info(f"Step {global_step}: eval_reward = {eval_reward:.4f}")

                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        self.save_checkpoint("best_model")

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")

            # End of epoch
            avg_epoch_reward = np.mean(epoch_rewards)
            logger.info(
                f"Epoch {epoch + 1} complete. Avg reward: {avg_epoch_reward:.4f}"
            )

        # Final save
        self.save_checkpoint("final_model")
        logger.info("Training complete!")

        return self.train_metrics

    def evaluate(self, eval_dataset: RoutingDataset) -> float:
        """Evaluate router on validation set."""
        self.model.eval()

        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )

        all_rewards = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                examples = batch["example"]

                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                for output, example in zip(outputs, examples):
                    generated = self.tokenizer.decode(output, skip_special_tokens=True)
                    predicted_model = self._extract_model(
                        generated, example["candidate_models"]
                    )

                    reward, _ = self.reward_model.compute_reward(
                        predicted_model=predicted_model,
                        optimal_model=example["optimal_model"],
                        generated_response=generated,
                        ground_truth=example.get("ground_truth", ""),
                    )
                    all_rewards.append(reward)

        return np.mean(all_rewards)

    def _extract_model(self, generated: str, candidates: List[str]) -> str:
        """Extract selected model from generated text."""
        # Look for model name after <route> tag
        generated_lower = generated.lower()

        for model in candidates:
            if model.lower() in generated_lower:
                return model

        # Default to first candidate if no match found
        return candidates[0] if candidates else "unknown"

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save config and metrics
        with open(os.path.join(save_path, "training_config.yaml"), "w") as f:
            yaml.dump(vars(self.config), f)

        with open(os.path.join(save_path, "training_metrics.json"), "w") as f:
            json.dump(self.train_metrics, f)

        logger.info(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Router-R1 model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/router_r1_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = RouterR1Config.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = RouterR1Config()

    # Apply overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = RouterR1Trainer(config)

    # Load datasets
    train_dataset = RoutingDataset(
        config.train_data_path,
        trainer.tokenizer,
        config.model_max_length,
    )

    eval_dataset = None
    if os.path.exists(config.eval_data_path):
        eval_dataset = RoutingDataset(
            config.eval_data_path,
            trainer.tokenizer,
            config.model_max_length,
        )

    logger.info(f"Training on {len(train_dataset)} examples")

    # Train
    metrics = trainer.train(train_dataset, eval_dataset)

    # Save final metrics
    with open(os.path.join(config.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f)

    logger.info(f"Training complete! Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
