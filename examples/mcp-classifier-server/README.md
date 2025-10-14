# MCP Classification Server

Example MCP servers that provide text classification with intelligent routing for the semantic router.

## 📦 Two Implementations

This directory contains **two MCP classification servers**:

### 1. **Regex-Based Server** (`server.py`)

- ✅ **Simple & Fast** - Pattern matching with regex
- ✅ **Lightweight** - ~10MB memory, <5ms per query
- ✅ **No Dependencies** - Just MCP SDK
- 📝 **Best For**: Prototyping, simple rules, low-latency requirements

### 2. **Embedding-Based Server** (`server_embedding.py`) 🆕

- ✅ **High Accuracy** - Semantic understanding with Qwen3-Embedding-0.6B
- ✅ **RAG-Style** - FAISS vector database with similarity search
- ✅ **Flexible** - Handles paraphrases, synonyms, variations
- 📝 **Best For**: Production use, high-accuracy requirements

**Choose based on your needs:**

- **Quick start / Testing?** → Use `server.py` (regex-based)
- **Production / Accuracy?** → Use `server_embedding.py` (embedding-based)

---

## Regex-Based Server (`server.py`)

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
python server.py --http --port 8090

# Stdio mode (for MCP clients)
python server.py
```

**Test the server:**

```bash
curl http://localhost:8090/health
# → {"status": "ok", "categories": ["math", "science", "technology", "history", "general"]}
```

## Configuration

**Router config (`config-mcp-classifier-example.yaml`):**

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
github.com/vllm-project/semantic-router/src/semantic-router/pkg/connectivity/mcp/api
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
- **FAISS vector database** for fast similarity search
- **RAG-style classification** using 95 training examples
- **Same MCP protocol** as regex server (drop-in replacement)
- **Higher accuracy** - Understands semantic meaning, not just patterns

### Comparison

| Feature | Regex (`server.py`) | Embedding (`server_embedding.py`) |
|---------|---------------------|-----------------------------------|
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Speed** | ~1-5ms | ~50-100ms |
| **Memory** | ~10MB | ~600MB |
| **Setup** | Simple | Requires model |
| **Best For** | Prototyping | Production |
