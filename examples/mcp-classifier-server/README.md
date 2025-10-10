# MCP Classification Server

Example MCP server that provides text classification with intelligent routing for the semantic router.

## Features

- **Dynamic Categories**: Loaded from MCP server at runtime via `list_categories`
- **Intelligent Routing**: Returns `model` and `use_reasoning` in classification response  
- **Regex-Based**: Simple pattern matching (replace with ML models for production)
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
   {"categories": ["math", "science", "technology", ...]}
   ```

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

Edit `CATEGORIES` to add categories:

```python
CATEGORIES = {
    "your_category": {
        "patterns": [r"\b(keyword1|keyword2)\b"],
        "description": "Your description"
    }
}
```

Edit `decide_routing()` for custom routing logic:

```python
def decide_routing(text, category, confidence):
    if category == "math":
        return "deepseek/deepseek-math", False
    return "openai/gpt-oss-20b", True
```

## License

MIT
