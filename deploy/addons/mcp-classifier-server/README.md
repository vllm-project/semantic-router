# MCP Classification Server

Example MCP servers that provide text classification with intelligent routing for the semantic router.

## 📦 Three Implementations

This directory contains **three MCP classification servers**:

### 1. **Regex-Based Server** (`server_keyword.py`)

- ✅ **Simple & Fast** - Pattern matching with regex
- ✅ **Lightweight** - ~10MB memory, <5ms per query
- ✅ **No Dependencies** - Just MCP SDK
- 📝 **Best For**: Prototyping, simple rules, low-latency requirements

### 2. **Embedding-Based Server** (`server_embedding.py`)

- ✅ **High Accuracy** - Semantic understanding with Qwen3-Embedding-0.6B
- ✅ **RAG-Style** - Milvus vector database with similarity search
- ✅ **Flexible** - Handles paraphrases, synonyms, variations
- 📝 **Best For**: Production use when you have good training examples

### 3. **Generative Model Server** (`server_generative.py`) 🆕

- ✅ **Highest Accuracy** - Fine-tuned Qwen3 generative model
- ✅ **True Probabilities** - Softmax-based probability distributions
- ✅ **Better Generalization** - Learns category patterns, not just examples
- ✅ **Entropy Calculation** - Shannon entropy for uncertainty quantification
- ✅ **HuggingFace Support** - Load models from HuggingFace Hub or local paths
- 📝 **Best For**: Production use with fine-tuned models (70-85% accuracy)

**Choose based on your needs:**

- **Quick start / Testing?** → Use `server_keyword.py` (regex-based)
- **Production with training examples?** → Use `server_embedding.py` (embedding-based)
- **Production with fine-tuned model?** → Use `server_generative.py` (generative model)

---

## Regex-Based Server (`server_keyword.py`)

### Features

- **Dynamic Categories**: Loaded from MCP server at runtime via `list_categories`
- **Per-Category System Prompts**: Each category has its own specialized system prompt for LLM context
- **Intelligent Routing**: Returns `model` and `use_reasoning` in classification response  
- **Regex-Based**: Simple pattern matching (fast but limited)
- **Dual Transport**: Supports both HTTP and stdio

## Categories

| Index | Category | Example Keywords |
|-------|----------|------------------|
| 0 | math | calculate, equation, formula, integral |
| 1 | science | physics, chemistry, biology, atom, DNA |
| 2 | technology | computer, programming, AI, cloud |
| 3 | history | ancient, war, empire, civilization |
| 4 | general | Catch-all for other queries |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# HTTP mode (for semantic router)
python server_keyword.py --http --port 8090

# Stdio mode (for MCP clients)
python server_keyword.py
```

**Test the server:**

```bash
curl http://localhost:8090/health
# → {"status": "ok", "categories": ["math", "science", "technology", "history", "general"]}
```

## Configuration

**Router config (`config-mcp-classifier.yaml`):**

```yaml
classifier:
  category_model:
    model_id: ""  # Empty = use MCP
  
  mcp_category_model:
    enabled: true
    transport_type: "http"
    url: "http://localhost:8090/mcp"
    # tool_name: optional - auto-discovers classification tool if not specified
    threshold: 0.6
    timeout_seconds: 30

categories: []  # Loaded dynamically from MCP
default_model: openai/gpt-oss-20b
```

**Tool Auto-Discovery:**
The router automatically discovers classification tools from the MCP server by:

1. Listing available tools on connection
2. Looking for common names: `classify_text`, `classify`, `categorize`, `categorize_text`
3. Pattern matching for tools containing "classif" in name/description
4. Optionally specify `tool_name` to use a specific tool

## Protocol API

This server implements the MCP classification protocol defined in:

```
github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcp/api
```

**Required Tools:**

1. **`list_categories`** - Returns `ListCategoriesResponse`:

   ```json
   {
     "categories": ["math", "science", "technology", "history", "general"],
     "category_system_prompts": {
       "math": "You are a mathematics expert. When answering math questions...",
       "science": "You are a science expert. When answering science questions...",
       "technology": "You are a technology expert. When answering tech questions..."
     },
     "category_descriptions": {
       "math": "Mathematical and computational queries",
       "science": "Scientific concepts and queries"
     }
   }
   ```

   The `category_system_prompts` and `category_descriptions` fields are optional but recommended.
   Per-category system prompts allow the MCP server to provide specialized instructions for each
   category that the router can inject when processing queries in that specific category.

2. **`classify_text`** - Returns `ClassifyResponse`:

   ```json
   {
     "class": 1,
     "confidence": 0.85,
     "model": "openai/gpt-oss-20b",
     "use_reasoning": true
   }
   ```

See the `api` package for full type definitions and documentation.

## How It Works

**Intelligent Routing Rules:**

- Long query (>20 words) + complex words (`why`, `how`, `explain`) → `use_reasoning: true`
- Math + short query → `use_reasoning: false`  
- High confidence (>0.9) → `use_reasoning: false`
- Low confidence (<0.6) → `use_reasoning: true`
- Default → `use_reasoning: true`

## Customization

**Edit `CATEGORIES` to add categories with per-category system prompts:**

```python
CATEGORIES = {
    "your_category": {
        "patterns": [r"\b(keyword1|keyword2)\b"],
        "description": "Your description",
        "system_prompt": """You are an expert in your_category. When answering:
- Provide specific guidance
- Use domain-specific terminology
- Follow best practices for this domain"""
    }
}
```

Each category can have its own specialized system prompt tailored to that domain.

**Edit `decide_routing()` for custom routing logic:**

```python
def decide_routing(text, category, confidence):
    if category == "math":
        return "deepseek/deepseek-math", False
    return "openai/gpt-oss-20b", True
```

**Using Per-Category System Prompts in the Router:**

The router stores per-category system prompts when loading categories. To use them:

```go
// After classifying a query, get the category-specific system prompt
category := "math"  // from classification result
if systemPrompt, ok := classifier.GetCategorySystemPrompt(category); ok {
    // Inject the category-specific system prompt when making LLM requests
    // Each category gets its own specialized instructions
}
```

---

## Embedding-Based Server (`server_embedding.py`)

For **production use with high accuracy**, see the embedding-based server:

### Quick Start

```bash
# Install dependencies
pip install -r requirements_embedding.txt

# Start server (HTTP mode on port 8090)
python3 server_embedding.py --http --port 8090
```

### Features

- **Qwen3-Embedding-0.6B** model with 1024-dimensional embeddings
- **Milvus vector database** for fast similarity search
- **RAG-style classification** using 95 training examples
- **Same MCP protocol** as regex server (drop-in replacement)
- **Higher accuracy** - Understands semantic meaning, not just patterns

### Comparison

| Feature | Regex (`server_keyword.py`) | Embedding (`server_embedding.py`) | Generative (`server_generative.py`) |
|---------|---------------------|-----------------------------------|-------------------------------------|
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ~1-5ms | ~50-100ms | ~100-200ms (GPU) |
| **Memory** | ~10MB | ~600MB | ~2GB (GPU) / ~4GB (CPU) |
| **Setup** | Simple | CSV + embeddings | Fine-tuned model required |
| **Probabilities** | Rule-based | Similarity scores | Softmax (true) |
| **Entropy** | No | Manual calculation | Built-in (Shannon) |
| **Best For** | Prototyping | Examples-based production | Model-based production |

---

## Generative Model Server (`server_generative.py`)

For **production use with a fine-tuned model and highest accuracy**, see the generative model server.

### Quick Start

**Option 1: Use Pre-trained HuggingFace Model** (Easiest)

```bash
# Server automatically downloads from HuggingFace Hub
python server_generative.py --http --port 8092 --model-path llm-semantic-router/qwen3_generative_classifier_r16
```

**Option 2: Train Your Own Model**

Step 1: Train the model

```bash
cd ../../../src/training/training_lora/classifier_model_fine_tuning_lora/
python ft_qwen3_generative_lora.py --mode train --epochs 8 --lora-rank 16
# Creates: qwen3_generative_classifier_r16/
```

Step 2: Start the server

```bash
cd -  # Back to examples/mcp-classifier-server/
python server_generative.py --http --port 8092 --model-path ../../../src/training/training_lora/classifier_model_fine_tuning_lora/qwen3_generative_classifier_r16
```

### Features

- **Fine-tuned Qwen3-0.6B** generative model with LoRA
- **Softmax probabilities** from model logits (true probability distribution)
- **Shannon entropy** for uncertainty quantification
- **14 MMLU-Pro categories** (biology, business, chemistry, CS, economics, engineering, health, history, law, math, other, philosophy, physics, psychology)
- **Same MCP protocol** as other servers (drop-in replacement)
- **Highest accuracy** - 70-85% on validation set

### Why Use Generative Server?

**Advantages over Embedding Server:**

- ✅ True probability distributions (softmax-based, not similarity-based)
- ✅ Better generalization beyond training examples
- ✅ More accurate classification (70-85% vs ~60-70%)
- ✅ Built-in entropy calculation for uncertainty
- ✅ Fine-tuned on task-specific data

**When to Use:**

- You have training data to fine-tune a model
- Need highest accuracy for production
- Want true probability distributions
- Need uncertainty quantification (entropy)
- Can afford 2-4GB memory footprint

### Testing

Test the generative server with sample queries:

```bash
python test_generative.py --model-path qwen3_generative_classifier_r16
```

### Documentation

For detailed documentation, see [README_GENERATIVE.md](README_GENERATIVE.md).
