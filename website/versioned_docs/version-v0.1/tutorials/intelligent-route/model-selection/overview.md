# Model Selection Overview

Model selection is an advanced feature of vLLM Semantic Router that automatically chooses the best LLM from multiple candidates based on learned preferences, query similarity, and cost-quality optimization.

The semantic router supports **8 selection algorithms** across two categories:

- **Core algorithms**: Static, Elo, RouterDC, AutoMix, Hybrid
- **RL-driven algorithms**: Thompson Sampling, GMTRouter, Router-R1

## What Problem Does It Solve?

When you have multiple LLM backends (e.g., GPT-4, Claude, Llama, Mistral), you face a challenge: **which model should handle each request?**

Traditional approaches:

- **Static routing**: Always use the same model (simple but suboptimal)
- **Round-robin**: Distribute evenly (ignores model strengths)
- **Random**: No intelligence (wastes resources)

Model selection solves this by **intelligently matching queries to models** based on:

1. **Learned quality preferences** (Elo ratings from user feedback)
2. **Query-model similarity** (RouterDC embeddings)
3. **Cost-quality tradeoffs** (AutoMix optimization)
4. **Combined signals** (Hybrid approach)

## Available Algorithms

### Core Algorithms

| Algorithm | Best For | Key Benefit |
|-----------|----------|-------------|
| [**Static**](./static.md) | Simple deployments | Predictable, zero overhead |
| [**Elo**](./elo.md) | Learning from feedback | Adapts to user preferences |
| [**RouterDC**](./router-dc.md) | Query-model matching | Matches specialties to queries |
| [**AutoMix**](./automix.md) | Cost optimization | Balances quality and cost |
| [**Hybrid**](./hybrid.md) | Complex requirements | Combines all methods |

### RL-Driven Algorithms

| Algorithm | Best For | Key Benefit |
|-----------|----------|-------------|
| [**Thompson Sampling**](./thompson-sampling.md) | Exploration/exploitation | Bayesian adaptive learning |
| [**GMTRouter**](./gmtrouter.md) | Personalization | Per-user preference learning |
| [**Router-R1**](./router-r1.md) | Complex reasoning | LLM-powered routing decisions |

## Quick Start

### Basic Configuration (Per-Decision)

Model selection is configured per-decision, allowing different strategies for different query types:

```yaml
decisions:
  - name: tech
    description: "Technical queries"
    priority: 10
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "tech"
    modelRefs:
      - model: "llama3.2:3b"
      - model: "phi4"
      - model: "gemma3:27b"
    algorithm:
      type: "elo"  # Use Elo rating for this decision
      elo:
        k_factor: 32
        category_weighted: true
```

### Algorithm Types

#### Static (Default)
Uses the first model in `modelRefs`. No learning, fully deterministic.

```yaml
algorithm:
  type: "static"
```

#### Elo Rating
Learns from user feedback to rank models by quality.

```yaml
algorithm:
  type: "elo"
  elo:
    k_factor: 32
    storage_path: "/var/lib/vsr/elo.json"
```

#### RouterDC
Matches query embeddings to model descriptions.

```yaml
algorithm:
  type: "router_dc"
  router_dc:
    temperature: 0.07
    require_descriptions: true
```

#### AutoMix
Optimizes cost-quality tradeoff using POMDP.

```yaml
algorithm:
  type: "automix"
  automix:
    cost_quality_tradeoff: 0.4
```

#### Hybrid
Combines all methods with configurable weights.

```yaml
algorithm:
  type: "hybrid"
  hybrid:
    elo_weight: 0.3
    router_dc_weight: 0.3
    automix_weight: 0.2
    cost_weight: 0.2
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Query                                   │
│                    "Explain quantum computing"                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Decision Matching                               │
│                 Decision "tech" matches → 3 models                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Selection Algorithm                               │
│                                                                      │
│  algorithm.type: "elo"                                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ EloSelector.Select()                                    │        │
│  │                                                         │        │
│  │ Model Ratings:                                          │        │
│  │   llama3.2:3b  → 1468 (0 wins, 2 losses)               │        │
│  │   phi4         → 1501 (3 wins, 2 losses)               │        │
│  │   gemma3:27b   → 1531 (5 wins, 1 loss) ← HIGHEST       │        │
│  └─────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Selected: gemma3:27b                            │
│                   (highest Elo rating: 1531)                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Choosing an Algorithm

See [Choosing the Right Algorithm](./choosing-algorithm.md) for detailed guidance.

**Quick Decision Tree:**

1. **Just getting started?** → Use `static` (default)
2. **Have user feedback?** → Use `elo`
3. **Have model descriptions?** → Use `router_dc`
4. **Want cost optimization?** → Use `automix`
5. **Need everything?** → Use `hybrid`

## Related Features

- **User Feedback Routing** - Collect feedback signals via `/api/v1/feedback` endpoint
- **Preference Routing** - Route based on user preferences in the system
- **Domain Routing** - Route by topic category using embedding classification

## Reference Papers

The selection algorithms are based on these research papers:

### Core Algorithms

- **Elo**: [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665) (ICLR 2025) - 85% cost reduction, 95% GPT-4 performance
- **RouterDC**: [Query-Based Router by Dual Contrastive Learning](https://arxiv.org/abs/2409.19886) (NeurIPS 2024) - +2.76% accuracy improvement
- **AutoMix**: [Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963) (NeurIPS 2024) - >50% cost reduction
- **Hybrid**: [Cost-Efficient Quality-Aware Query Routing](https://arxiv.org/abs/2404.14618) (ICLR 2024) - 40% fewer expensive calls

### RL-Driven Algorithms

- **Thompson Sampling**: Classical multi-armed bandit approach (Agrawal & Goyal) - Bayesian exploration/exploitation
- **GMTRouter**: [Personalized LLM Routing via Graph Neural Networks](https://arxiv.org/abs/2511.08590) (arXiv) - 0.9-21.6% accuracy improvement
- **Router-R1**: [Teaching LLMs Multi-Round Routing via RL](https://arxiv.org/abs/2506.09033) (NeurIPS 2025) - Outperforms single-round baselines
