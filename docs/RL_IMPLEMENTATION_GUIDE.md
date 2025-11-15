# Reinforcement Learning Implementation Guide for Intent Classifier

## Overview

This document outlines how Reinforcement Learning (RL) can be integrated into the intent classifier training pipeline. The existing architecture uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning; RL will layer on top of this to optimize policies based on reward signals.

## Current Architecture

### Rust Components (`candle-binding/`)

- **`IntentLoRAClassifier`** (`src/classifiers/lora/intent_lora.rs`): Inference engine using merged LoRA models
  - Loads frozen BERT backbone + LoRA adapters
  - Classifies text to one of N intent categories
  - Returns: intent label, confidence score, processing time
  - Methods: `classify_intent()`, `batch_classify()`, `parallel_classify()`

- **`HighPerformanceBertClassifier`** (`src/model_architectures/lora/bert_lora.rs`): Low-level model
  - Manages frozen BERT embeddings + LoRA parameter matrices (B, A)
  - Runs forward pass: embeddings → BERT layers → pooling → classification head
  - Outputs logits → softmax → (class_idx, confidence)

### Python Components (`src/training/training_lora/`)

- **`ft_linear_lora.py`**: LoRA fine-tuning using supervised loss (cross-entropy)
  - Trains on labeled MMLU-Pro dataset (14 categories)
  - Optimizes model to predict correct intent from text
  - Merges LoRA adapter with base model for Rust inference

- **`rl_utils.py`**: Config loading (NEW)
  - Reads `classifier.rl_training` from `config/config.yaml`
  - Exposes RL hyperparams: `algorithm`, `learning_rate`, `gamma`, `batch_size`, `update_epochs`, `reward_metric`

## How RL Can Be Implemented

### 1. **Reward Function Design**

RL needs a reward signal that reflects model performance. Options:

#### A. **Task-Based Reward** (Recommended for Intent Classification)
```python
def compute_reward(predictions, labels, metric="accuracy"):
    """
    Compute immediate reward from model predictions.
    
    Args:
        predictions: (batch_size,) predicted class indices
        labels: (batch_size,) ground truth labels
        metric: "accuracy" | "f1" | "precision" | "recall"
    
    Returns:
        reward: scalar in [0, 1] or [-1, 1]
    """
    if metric == "accuracy":
        # Sparse: 1.0 if correct, 0.0 if incorrect
        reward = (predictions == labels).mean().item()
    elif metric == "f1":
        # Use sklearn.metrics.f1_score()
        reward = f1_score(labels, predictions, average="weighted")
    elif metric == "confidence":
        # Weighted by model confidence
        correct = (predictions == labels)
        confidence = softmax(logits, dim=1).max(dim=1)[0]
        reward = (correct.float() * confidence).mean()
    
    # Normalize to [-1, 1] for RL algorithms
    return 2.0 * reward - 1.0
```

#### B. **Calibration Reward** (For Uncertainty Quantification)
```python
def compute_calibration_reward(confidences, predictions, labels):
    """
    Reward high confidence on correct predictions, low confidence on incorrect.
    """
    correct = (predictions == labels).float()
    # Expected Calibration Error-style reward
    calibration_gap = (confidences - correct).abs()
    return 1.0 - calibration_gap.mean()
```

#### C. **Latency-Aware Reward** (For Real-Time Systems)
```python
def compute_latency_aware_reward(accuracy, latency_ms, target_latency=100):
    """
    Balance accuracy vs. inference speed.
    """
    latency_penalty = max(0, (latency_ms - target_latency) / target_latency)
    return accuracy * (1.0 - 0.5 * latency_penalty)
```

### 2. **RL Algorithm Integration Points**

#### **Option A: Policy Gradient (PPO/A2C)** — Recommended

**Architecture:**
```
Supervised LoRA Model (pre-trained)
         ↓
    RL Policy Head (learned via PPO)
         ↓
    Action: select confidence threshold or prediction adjustment
    Reward: task-specific metric (accuracy, F1, etc.)
```

**Implementation Steps:**

1. **Start with Supervised LoRA Model** as initialization
   ```python
   # Load trained supervised LoRA model from ft_linear_lora.py
   model = PeftModel.from_pretrained(base_model, lora_adapter_path)
   initial_policy = model  # LoRA weights = initial policy
   ```

2. **Collect Rollouts** (on-policy data)
   ```python
   def collect_rollout(policy, train_loader, horizon=1000):
       """
       Run policy on training data, collect (state, action, reward) tuples.
       """
       trajectories = []
       for batch_idx, (texts, labels) in enumerate(train_loader):
           # Forward pass
           logits = policy(texts)  # (batch_size, num_classes)
           confidences = softmax(logits, dim=1)
           predictions = argmax(logits, dim=1)
           
           # Compute reward
           reward = compute_reward(predictions, labels, metric)
           
           # Store trajectory
           trajectories.append({
               "text": texts,
               "logits": logits,
               "reward": reward,
               "label": labels
           })
           
           if batch_idx >= horizon:
               break
       
       return trajectories
   ```

3. **Compute Advantages & Update Policy (PPO)**
   ```python
   def ppo_update(policy, trajectories, config):
       """
       PPO update step: optimize policy to maximize advantage-weighted log-probs.
       """
       advantages = compute_gae(trajectories, config.gamma, config.lambda)
       
       for epoch in range(config.update_epochs):
           for batch in minibatch(trajectories, config.batch_size):
               # Forward pass
               new_logits = policy(batch["text"])
               new_log_probs = log_softmax(new_logits, dim=1)
               old_log_probs = log_softmax(batch["logits"], dim=1)
               
               # PPO loss: clip probability ratio
               ratio = exp(new_log_probs - old_log_probs)
               clipped_ratio = clamp(ratio, 1 - config.eps_clip, 1 + config.eps_clip)
               ppo_loss = -min(ratio * advantages, clipped_ratio * advantages).mean()
               
               # Update LoRA weights
               ppo_loss.backward()
               optimizer.step()
   ```

**Code Location:** Create `src/training/training_lora/rl_ppo_trainer.py`

#### **Option B: DQN (Q-Learning)** — For Discrete Action Space

Use DQN if actions are discrete (e.g., confidence threshold selection):

```python
class IntentRLAgent:
    def __init__(self, model, num_actions=5):
        """
        num_actions: discrete actions like [threshold_0.5, 0.6, 0.7, 0.8, 0.9]
        """
        self.model = model
        self.q_network = QNetwork(model.hidden_size, num_actions)
        self.target_network = copy.deepcopy(self.q_network)
        
    def forward(self, text):
        # Get intent classification logits
        logits = self.model(text)  # (num_classes,)
        confidence = max(softmax(logits))
        
        # Get Q-values for action selection (which threshold to use)
        q_values = self.q_network(logits)  # (num_actions,)
        action = argmax(q_values)
        
        return action, confidence
    
    def update(self, state, action, reward, next_state, done):
        """DQN Bellman update"""
        q_pred = self.q_network(state)[action]
        q_target = reward + gamma * max(self.target_network(next_state))
        loss = (q_pred - q_target) ** 2
        loss.backward()
```

**Code Location:** Create `src/training/training_lora/rl_dqn_trainer.py`

### 3. **Integration with Existing Training Pipeline**

#### **Modify `ft_linear_lora.py`:**

```python
def main(..., enable_rl=False, rl_algorithm="ppo", ...):
    """Train with optional RL fine-tuning."""
    
    # Phase 1: Supervised LoRA training (existing)
    model, tokenizer = train_supervised_lora(
        model_name, 
        train_dataset, 
        val_dataset,
        num_epochs=num_epochs
    )
    
    # Phase 2: Optional RL fine-tuning on top
    if enable_rl:
        logger.info(f"Starting RL fine-tuning with {rl_algorithm}...")
        
        # Load RL config from YAML
        rl_config = load_rl_config()
        
        if rl_algorithm == "ppo":
            from rl_ppo_trainer import PPOTrainer
            rl_trainer = PPOTrainer(model, rl_config)
        elif rl_algorithm == "dqn":
            from rl_dqn_trainer import DQNTrainer
            rl_trainer = DQNTrainer(model, rl_config)
        
        # Collect rollouts and update policy
        for epoch in range(rl_config["update_epochs"]):
            trajectories = collect_rollout(model, train_loader)
            rl_trainer.update(trajectories)
        
        # Evaluate RL model
        rl_val_metrics = evaluate_rl_model(model, val_dataset, rl_config)
        logger.info(f"RL Validation: {rl_val_metrics}")
    
    # Save final model
    model.save_pretrained(output_dir)
```

#### **CLI Integration:**

```python
parser.add_argument("--enable-rl", action="store_true", 
                    help="Enable RL fine-tuning after supervised training")
parser.add_argument("--rl-algorithm", choices=["ppo", "a2c", "dqn"], 
                    default="ppo", help="RL algorithm to use")
parser.add_argument("--rl-epochs", type=int, default=4, 
                    help="Number of RL update epochs")
parser.add_argument("--rl-reward-metric", 
                    choices=["accuracy", "f1", "calibration"], 
                    default="accuracy", help="Reward signal metric")
```

### 4. **Rust Integration (Runtime)**

#### **Add RL Policy Head to `IntentLoRAClassifier`:**

```rust
// In intent_lora.rs

pub struct IntentRLClassifier {
    // Existing supervised model
    supervised_classifier: IntentLoRAClassifier,
    
    // Optional RL components (loaded if available)
    rl_policy_head: Option<Arc<HighPerformanceBertClassifier>>,
    rl_config: Option<RLConfig>,
    
    // Confidence adjustment learned by RL
    confidence_adjustment: f32,
}

impl IntentRLClassifier {
    pub fn new(model_path: &str, use_rl: bool, use_cpu: bool) -> Result<Self> {
        let supervised = IntentLoRAClassifier::new(model_path, use_cpu)?;
        
        // Load RL config if enabled
        let rl_config = if use_rl {
            use crate::core::config_loader::GlobalConfigLoader;
            Some(GlobalConfigLoader::load_classifier_rl_config_safe())
        } else {
            None
        };
        
        // Load RL policy head if available
        let rl_policy_head = if use_rl {
            // Try to load from model_path/rl_policy_head.safetensors
            Self::load_rl_policy_head(model_path, use_cpu).ok()
        } else {
            None
        };
        
        Ok(Self {
            supervised_classifier: supervised,
            rl_policy_head,
            rl_config,
            confidence_adjustment: 1.0,
        })
    }
    
    pub fn classify_intent_with_rl(&self, text: &str) -> Result<IntentResult> {
        let mut result = self.supervised_classifier.classify_intent(text)?;
        
        // If RL policy available, adjust confidence
        if let Some(rl_head) = &self.rl_policy_head {
            result.confidence *= self.confidence_adjustment;
        }
        
        Ok(result)
    }
    
    fn load_rl_policy_head(model_path: &str, use_cpu: bool) -> Result<Arc<HighPerformanceBertClassifier>> {
        // Load from safetensors or similar
        // Return: RL-tuned policy head
        todo!("Implement RL policy head loading")
    }
}
```

### 5. **Test Integration**

#### **Add RL-specific tests to `intent_lora_test.rs`:**

```rust
#[test]
fn test_intent_rl_policy_reward_signal() {
    // Verify reward computation matches expected metric
    let predictions = vec![0, 1, 2];
    let labels = vec![0, 0, 2];
    let reward = compute_reward(&predictions, &labels, "accuracy");
    assert_eq!(reward, 2.0 / 3.0);  // 2 correct out of 3
}

#[test]
#[serial]
fn test_intent_rl_policy_update() {
    let classifier = IntentRLClassifier::new(MODEL_PATH, true, true).unwrap();
    
    // Collect trajectory
    let texts = vec!["hello", "goodbye", "how are you"];
    let labels = vec![0, 1, 0];
    
    // Run through RL classifier
    let results = classifier.batch_classify(&texts).unwrap();
    
    // Verify RL adjustment applied
    for result in &results {
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
    }
}
```

### 6. **Config-Driven RL Enablement**

Update `config/config.yaml`:

```yaml
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"
  
  rl_training:
    enabled: false            # Set to true to enable RL
    algorithm: "ppo"          # "ppo" | "a2c" | "dqn"
    learning_rate: 1e-05
    gamma: 0.99
    batch_size: 16
    update_epochs: 4
    reward_metric: "accuracy" # "accuracy" | "f1" | "calibration"
    
    # Optional RL-specific tuning
    ppo_eps_clip: 0.2         # PPO clipping parameter
    ppo_lambda: 0.95          # GAE lambda for advantage estimation
    dqn_epsilon: 0.1          # DQN exploration rate
    calibration_target: 0.95  # For calibration reward
```

## Implementation Roadmap

### Phase 1: Foundation (Already Done)
- ✅ Config schema in `config.yaml`
- ✅ Rust `RLConfig` loader
- ✅ Python `rl_utils.py` helper
- ✅ Integration points in `ft_linear_lora.py` (logs RL config)

### Phase 2: Core RL (Next)
- [ ] Implement `rl_ppo_trainer.py` with on-policy updates
- [ ] Add reward function implementations (accuracy, F1, calibration)
- [ ] Integrate PPO into `ft_linear_lora.py` main training loop
- [ ] Add RL evaluation metrics (episodic return, advantage, loss)

### Phase 3: Runtime Integration
- [ ] Load RL policy head in Rust (`intent_lora.rs`)
- [ ] Route inference through RL classifier when enabled
- [ ] Add RL-specific telemetry (confidence adjustment, policy entropy)

### Phase 4: Advanced Features
- [ ] Multi-task RL (simultaneous intent + PII + security)
- [ ] Batch adaptation (update policy per batch)
- [ ] Curriculum learning (easy → hard examples)
- [ ] Meta-RL for few-shot intent adaptation

## Detailed Implementation for PPO (Start Here)

See **`RL_PPO_IMPLEMENTATION.md`** (to be created) for step-by-step PPO trainer code.

## References

- **PPO Paper:** Schulman et al., "Proximal Policy Optimization Algorithms" (https://arxiv.org/abs/1707.06347)
- **Reward Shaping:** Ng et al., "Policy Invariance Under Reward Transformations" (https://arxiv.org/abs/2103.01808)
- **LoRA + RL:** QLoRA for efficient fine-tuning + RL (https://arxiv.org/abs/2305.14314)

---

**Next Steps:**
1. Review this design with team
2. Start Phase 2 implementation with `rl_ppo_trainer.py`
3. Add tests for reward functions
4. Benchmark RL-trained models vs. supervised baselines
