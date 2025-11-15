"""
PPO (Proximal Policy Optimization) Trainer for Intent Classification RL

Implements on-policy RL fine-tuning on top of supervised LoRA models.
Uses collected rollouts to optimize the policy via PPO loss.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PPOBuffer:
    """Experience buffer for PPO training."""

    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.texts = []
        self.logits = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(
        self,
        text: str,
        logits: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        value: float,
    ):
        """Add experience to buffer."""
        if len(self.texts) >= self.capacity:
            # Fifo buffer
            self.texts.pop(0)
            self.logits.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.values.pop(0)

        self.texts.append(text)
        self.logits.append(logits.detach().cpu())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get_batch(self, batch_size: int) -> Tuple[List, List, List, List, List]:
        """Sample random batch from buffer."""
        if len(self.texts) < batch_size:
            indices = list(range(len(self.texts)))
        else:
            indices = np.random.choice(len(self.texts), batch_size, replace=False)

        return (
            [self.texts[i] for i in indices],
            [self.logits[i] for i in indices],
            [self.actions[i] for i in indices],
            [self.rewards[i] for i in indices],
            [self.values[i] for i in indices],
        )

    def compute_advantages(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[List, List]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []

        next_value = 0.0
        gae = 0.0

        # Reverse iteration for GAE computation
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = self.values[t + 1]

            delta = (
                self.rewards[t] + gamma * next_value_t * next_non_terminal
            ) - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        # Normalize advantages
        advantages = np.array(advantages)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.tolist(), returns


class PPOTrainer:
    """PPO trainer for RL fine-tuning on intent classification."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        batch_size: int = 16,
        update_epochs: int = 4,
    ):
        """
        Initialize PPO trainer.

        Args:
            model: LoRA fine-tuned BERT model
            tokenizer: Tokenizer for model
            device: "cuda" or "cpu"
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda
            eps_clip: PPO clipping parameter
            entropy_coef: Coefficient for entropy regularization
            value_coef: Coefficient for value function loss
            batch_size: Batch size for updates
            update_epochs: Number of update epochs per rollout
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Buffer
        self.buffer = PPOBuffer(capacity=2000)

        # Metrics
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "episodic_return": [],
        }

        logger.info(f"PPO Trainer initialized with lr={learning_rate}, gamma={gamma}")

    def collect_rollout(
        self,
        train_loader: DataLoader,
        num_steps: Optional[int] = None,
        reward_metric: str = "accuracy",
    ) -> float:
        """
        Collect trajectories by running policy on training data.

        Args:
            train_loader: DataLoader with (texts, labels) tuples
            num_steps: Max number of steps to collect (None = full epoch)
            reward_metric: "accuracy" | "f1" | "calibration"

        Returns:
            Average episodic return
        """
        self.model.eval()
        self.buffer.clear()

        total_reward = 0.0
        num_samples = 0
        step = 0

        with torch.no_grad():
            for batch_idx, (texts, labels) in enumerate(train_loader):
                if num_steps is not None and step >= num_steps:
                    break

                # Forward pass
                batch_texts = texts if isinstance(texts, list) else texts.tolist()
                batch_labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

                encodings = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                outputs = self.model(**encodings)
                logits = outputs.logits  # (batch_size, num_classes)
                confidences = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                batch_labels = batch_labels.to(self.device)

                # Compute reward
                if reward_metric == "accuracy":
                    correct = (predictions == batch_labels).float()
                    reward = correct
                elif reward_metric == "f1":
                    # Simplified: use confidence as proxy for F1
                    max_confidence = confidences.max(dim=1)[0]
                    correct = (predictions == batch_labels).float()
                    reward = correct * max_confidence
                elif reward_metric == "calibration":
                    # Reward: high conf on correct, low conf on incorrect
                    correct = (predictions == batch_labels).float()
                    max_confidence = confidences.max(dim=1)[0]
                    calibration = correct * max_confidence + (1 - correct) * (1 - max_confidence)
                    reward = calibration
                else:
                    reward = (predictions == batch_labels).float()

                # Value function: estimate of expected future reward
                value = reward  # Simple baseline: immediate reward

                # Store in buffer
                for i, (text, logit, pred, rew, val) in enumerate(
                    zip(batch_texts, logits, predictions, reward, value)
                ):
                    self.buffer.add(
                        text=text,
                        logits=logit.detach().cpu(),
                        action=pred.item(),
                        reward=rew.item(),
                        done=(batch_idx == len(train_loader) - 1),  # Episode boundary
                        value=val.item() if isinstance(val, torch.Tensor) else val,
                    )
                    total_reward += rew.item()
                    num_samples += 1

                step += 1

        avg_return = total_reward / max(num_samples, 1)
        self.metrics["episodic_return"].append(avg_return)

        logger.info(
            f"Collected rollout: {num_samples} steps, avg return={avg_return:.4f}, reward_metric={reward_metric}"
        )

        return avg_return

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected trajectories.

        Returns:
            Dictionary with loss metrics
        """
        if len(self.buffer.texts) == 0:
            logger.warning("No trajectories in buffer, skipping update")
            return {}

        self.model.train()

        # Compute advantages
        advantages, returns = self.buffer.compute_advantages(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        epoch_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

        # Update epochs
        for epoch in range(self.update_epochs):
            # Mini-batch updates
            indices = np.arange(len(self.buffer.texts))
            np.random.shuffle(indices)

            for batch_start in range(0, len(indices), self.batch_size):
                batch_indices = indices[
                    batch_start : batch_start + self.batch_size
                ]

                batch_texts = [self.buffer.texts[i] for i in batch_indices]
                batch_old_logits = [
                    self.buffer.logits[i] for i in batch_indices
                ]
                batch_advantages = [advantages[i] for i in batch_indices]
                batch_returns = [returns[i] for i in batch_indices]

                # Forward pass
                encodings = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                outputs = self.model(**encodings)
                logits = outputs.logits

                # Get old log probabilities
                old_logits = torch.stack(batch_old_logits).to(self.device)
                old_log_probs = F.log_softmax(old_logits, dim=1)
                new_log_probs = F.log_softmax(logits, dim=1)

                # Value prediction
                values = logits.mean(dim=1)  # Simple value: mean logit

                # PPO loss
                advantages_t = torch.tensor(
                    batch_advantages, dtype=torch.float32, device=self.device
                )
                returns_t = torch.tensor(
                    batch_returns, dtype=torch.float32, device=self.device
                )

                # Probability ratio
                log_prob_ratio = new_log_probs.mean(dim=1) - old_log_probs.mean(dim=1)
                ratio = torch.exp(log_prob_ratio)

                # Clipped surrogate objective
                surr1 = ratio * advantages_t
                surr2 = (
                    torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages_t
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns_t)

                # Entropy (regularization)
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

                # Total loss
                total_loss = (
                    policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_metrics["policy_loss"].append(policy_loss.item())
                epoch_metrics["value_loss"].append(value_loss.item())
                epoch_metrics["entropy"].append(entropy.item())

        # Average metrics
        avg_metrics = {
            k: np.mean(v) if v else 0.0 for k, v in epoch_metrics.items()
        }
        for k, v in avg_metrics.items():
            self.metrics[k].append(v)

        logger.info(
            f"PPO Update: policy_loss={avg_metrics['policy_loss']:.4f}, "
            f"value_loss={avg_metrics['value_loss']:.4f}, "
            f"entropy={avg_metrics['entropy']:.4f}"
        )

        return avg_metrics

    def train_episode(
        self,
        train_loader: DataLoader,
        reward_metric: str = "accuracy",
        num_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run one full PPO episode: collect rollout + update policy.

        Args:
            train_loader: Training data
            reward_metric: Reward signal metric
            num_steps: Max steps in rollout

        Returns:
            Metrics dictionary
        """
        # Collect trajectories
        episodic_return = self.collect_rollout(
            train_loader, num_steps=num_steps, reward_metric=reward_metric
        )

        # Update policy
        update_metrics = self.update()
        update_metrics["episodic_return"] = episodic_return

        return update_metrics

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics."""
        return self.metrics


def train_with_rl(
    model,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_episodes: int = 5,
    reward_metric: str = "accuracy",
    device: str = "cuda",
    **ppo_kwargs,
) -> Dict:
    """
    Train model with PPO RL fine-tuning.

    Args:
        model: Supervised LoRA model
        tokenizer: Tokenizer
        train_loader: Training data
        val_loader: Validation data
        num_episodes: Number of PPO episodes
        reward_metric: Reward metric
        device: Device to use
        **ppo_kwargs: Additional PPO hyperparameters

    Returns:
        Training metrics dictionary
    """
    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        **ppo_kwargs,
    )

    best_return = -np.inf
    all_metrics = []

    for episode in range(num_episodes):
        logger.info(f"PPO Episode {episode + 1}/{num_episodes}")

        # Train episode
        metrics = trainer.train_episode(
            train_loader, reward_metric=reward_metric
        )
        all_metrics.append(metrics)

        # Validation (optional)
        if val_loader is not None:
            logger.info("Validating...")
            # Could add validation logic here
            pass

        # Track best
        if metrics["episodic_return"] > best_return:
            best_return = metrics["episodic_return"]
            logger.info(f"New best return: {best_return:.4f}")

    logger.info(f"PPO training complete. Best return: {best_return:.4f}")

    return {
        "best_episodic_return": best_return,
        "all_metrics": all_metrics,
        "final_metrics": trainer.get_metrics(),
    }
