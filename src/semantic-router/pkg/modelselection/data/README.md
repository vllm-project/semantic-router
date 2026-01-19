# Model Selection Training Data

## Files Overview

| File | Size | Records | Description |
|------|------|---------|-------------|
| `extended_training_data.jsonl` | 26 MB | 50,544 | **USE THIS** - All models including GPT/Claude |
| `training_data.jsonl` | 58 MB | 50,544 | Original training data (NVIDIA models only) |
| `test_data.jsonl` | 7.4 MB | 6,354 | Test data for evaluation |
| `llm_candidates.json` | 8 KB | 10 | Model configurations |
| `custom_training_template.jsonl` | 1 KB | 3 | Template for custom models |

## Models Included

### Extended Training Data (`extended_training_data.jsonl`)

| Model | Provider | Type |
|-------|----------|------|
| gpt-4 | OpenAI | Strong |
| gpt-4-turbo | OpenAI | Strong |
| gpt-3.5-turbo | OpenAI | Fast |
| claude-3-opus | Anthropic | Strong |
| claude-3-sonnet | Anthropic | Medium |
| claude-3-haiku | Anthropic | Fast |
| llama-3.1-8b-instruct | Meta/NVIDIA | Fast |
| llama3-chatqa-1.5-8b | NVIDIA | Fast |
| llama3-chatqa-1.5-70b | NVIDIA | Strong |
| mistral-7b-instruct | Mistral | Fast |
| qwen2.5-7b-instruct | Qwen | Medium |
| + more NVIDIA models... | | |

## Data Format

Each line in the JSONL file:

```json
{
  "query": "Write a Python function to sort a list",
  "task_name": "humaneval",
  "query_type": "coding",
  "model_scores": {
    "gpt-4": 0.92,
    "claude-3-opus": 0.90,
    "gpt-3.5-turbo": 0.70,
    "llama-3.1-8b": 0.68
  },
  "best_model": "gpt-4"
}
```

## Adding Custom LLM Models

### Step 1: Create Training Data

Create a JSONL file with your model's performance:

```json
{"query": "Your test query 1", "query_type": "coding", "model_scores": {"your-model": 0.85, "gpt-4": 0.90}, "best_model": "gpt-4"}
{"query": "Your test query 2", "query_type": "math", "model_scores": {"your-model": 0.92, "gpt-4": 0.88}, "best_model": "your-model"}
```

### Step 2: Add Model Config

Add your model to `llm_candidates.json`:

```json
{
  "your-custom-model": {
    "provider": "your-provider",
    "model_id": "your-model-id",
    "display_name": "Your Custom Model",
    "category": "strong",
    "cost_per_1k_input_tokens": 0.01,
    "cost_per_1k_output_tokens": 0.02,
    "max_context_length": 8192,
    "strengths": ["coding", "math"],
    "avg_latency_ms": 500,
    "quality_score": 0.85
  }
}
```

### Step 3: Train

```go
// Load base training data
baseData := LoadTrainingData("extended_training_data.jsonl")

// Load your custom data
customData := LoadTrainingData("custom_training_data.jsonl")

// Merge
allData := append(baseData, customData...)

// Train
selector := NewKNNSelector(5)
selector.Train(allData)
selector.Save("models/knn_with_custom.json")
```

## Query Types

| Type | Description | Example |
|------|-------------|---------|
| `math` | Mathematical problems | "Solve: 2x + 5 = 15" |
| `coding` | Programming tasks | "Write a sort function" |
| `reasoning` | Logic and analysis | "If all A are B..." |
| `writing` | Creative content | "Write a poem about..." |
| `simple` | Quick answers | "What is the capital of France?" |

## License

See `LICENSE` file. Training data derived from public benchmark datasets (MIT License).
