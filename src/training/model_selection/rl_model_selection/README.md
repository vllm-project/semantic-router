# RL-Driven Model Selection Training

This directory contains training scripts for RL-driven model selection, implementing the approaches from:

- **Router-R1** ([arXiv:2506.09033](https://arxiv.org/abs/2506.09033)) - Multi-round routing with RL training
- **GMTRouter** ([arXiv:2511.08590](https://arxiv.org/abs/2511.08590)) - Graph-based personalized routing

## Overview

The training pipeline teaches a router model to make intelligent routing decisions based on:

1. Query content and complexity
2. Model capabilities and costs
3. User preferences (personalization)
4. Historical performance data

## Directory Structure

```
rl_model_selection/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── configs/
│   ├── router_r1_config.yaml   # Router-R1 training config
│   └── gmtrouter_config.yaml   # GMTRouter training config
├── data/
│   ├── download_datasets.py    # Download NQ, HotpotQA
│   ├── prepare_training_data.py # Format for RL training
│   └── sample_routing_data.json # Sample training format
├── models/
│   ├── router_policy.py        # Router policy network
│   └── graph_router.py         # GMTRouter GNN model
├── train_router_r1.py          # Main Router-R1 training script
├── train_gmtrouter.py          # GMTRouter GNN training
├── evaluate_router.py          # Evaluation on benchmark
└── scripts/
    ├── train_router_r1_gpu.sh  # GPU training script
    └── train_gmtrouter_gpu.sh  # GMTRouter GPU script
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Training Data

```bash
python data/download_datasets.py --datasets nq hotpotqa
```

### 3. Train Router-R1 Policy

```bash
# On GPU machine (A100 recommended)
./scripts/train_router_r1_gpu.sh

# Or manually:
python train_router_r1.py \
    --config configs/router_r1_config.yaml \
    --output_dir ./checkpoints/router_r1 \
    --num_epochs 10 \
    --batch_size 32
```

### 4. Train GMTRouter Graph Model

```bash
./scripts/train_gmtrouter_gpu.sh
```

## Training Data Format

### Router-R1 Format

```json
{
  "query": "What is the capital of France?",
  "query_type": "factual",
  "candidate_models": [
    {"name": "gpt-4", "cost": 30.0, "latency": 0.5},
    {"name": "mistral-7b", "cost": 0.5, "latency": 0.1}
  ],
  "ground_truth": "Paris",
  "optimal_model": "mistral-7b",
  "reasoning": "Simple factual query, cheaper model sufficient"
}
```

### GMTRouter Format

```json
{
  "user_id": "user-123",
  "interactions": [
    {
      "query": "Help me debug this Python code",
      "model_used": "deepseek-coder",
      "user_feedback": "positive",
      "timestamp": 1706300000
    }
  ],
  "preferred_models": ["deepseek-coder", "gpt-4"],
  "context": "software developer, prefers detailed explanations"
}
```

## Hardware Requirements

| Training Type | GPU Memory | Training Time |
|--------------|------------|---------------|
| Router-R1 (7B router) | 24GB+ | 4-8 hours |
| GMTRouter GNN | 16GB+ | 2-4 hours |

## Pretrained Checkpoints

After training, checkpoints are available at:

- Router-R1: `./checkpoints/router_r1/best_model.pt`
- GMTRouter: `./checkpoints/gmtrouter/best_model.pt`

## Integration with VSR

The trained models integrate with the `RLDrivenSelector` in VSR:

```yaml
decisions:
  - name: intelligent_routing
    algorithm:
      type: "rl_driven"
      rl_driven:
        # Use trained Router-R1 model for routing decisions
        router_model_path: "/path/to/router_r1/best_model.pt"
        use_thompson_sampling: true
        enable_personalization: true
```

## References

1. Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning (NeurIPS 2025)
2. GMTRouter: Personalized LLM Router over Multi-turn User Interactions (arXiv 2511.08590)
