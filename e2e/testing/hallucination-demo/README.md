# Hallucination Detection Demo

End-to-end demo for testing the hallucination detection workflow with real tool calling.

## Overview

This demo creates a complete pipeline to demonstrate hallucination detection:

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Chat Client    │ --> │  Semantic Router │ --> │   Mock vLLM       │
│  (tool calling) │     │  (hallucination  │     │  (tool calls +    │
│                 │ <-- │   detection)     │ <-- │   hallucinated    │
│                 │     │                  │     │   responses)      │
└────────┬────────┘     └──────────────────┘     └───────────────────┘
         │
         v
┌─────────────────┐
│  Mock Web Search│
│  (ground truth  │
│   context)      │
└─────────────────┘
```

## Flow

1. **User asks a question** (e.g., "When was the Eiffel Tower built?")
2. **Mock vLLM returns `tool_calls`** to invoke `web_search`
3. **Chat client executes mock web search** → gets ground truth context
4. **Client sends tool results back** to the router
5. **Mock vLLM returns hallucinated response** (facts that conflict with context)
6. **Router's hallucination detector** catches the inconsistencies
7. **Response includes hallucination detection headers**

## Quick Start

```bash
# Run the complete demo
make demo-hallucination

# Or manually:
./e2e/testing/hallucination-demo/run_demo.sh
```

## Components

### mock_vllm_toolcall.py

Mock LLM server that:

- Returns `tool_calls` on first request (to invoke web_search)
- Returns hallucinated responses on follow-up (with tool results)

### mock_web_search.py

Mock search service that returns ground truth context:

- Eiffel Tower facts
- Tokyo population data
- Apple Inc. founding info

### chat_client.py

Interactive CLI client that:

- Sends questions through the semantic router
- Handles tool calls automatically
- Displays hallucination detection headers

## Demo Questions

Try these questions to see hallucination detection in action:

| Question | Ground Truth | Hallucinated Claim |
|----------|--------------|-------------------|
| When was the Eiffel Tower built? | 1889 | 1887, originally painted red |
| What is the population of Tokyo? | ~14 million | 45 million |
| Who founded Apple? | Jobs, Wozniak, Wayne | Includes Bill Gates |

## Requirements

- Python 3.8+
- requests library
- Semantic router built (`make build-router`)
- Hallucination detection models (`make download-models`)

## Manual Testing

```bash
# Start services individually
python3 e2e/testing/hallucination-demo/mock_vllm_toolcall.py --port 8002
python3 e2e/testing/hallucination-demo/mock_web_search.py --port 8003

# Start router
./bin/router -config=config/testing/config.hallucination.yaml

# Run client
python3 e2e/testing/hallucination-demo/chat_client.py \
    --router-url http://localhost:8801 \
    --search-url http://localhost:8003
```

## Expected Output

```
[Step 1] Sending question to router: When was the Eiffel Tower built?...
✓ Got 1 tool call(s)
[Step 2] Executing tool: web_search({'query': 'when was the eiffel tower built'})
✓ Got search context: 289 chars
   Context preview: The Eiffel Tower is located in Paris, France. Construction began in 1887...
[Step 3] Sending tool results to get final response...

────────────────────────────────────────────────────────────
LLM Response:
  The Eiffel Tower was built in 1887 by architect Gustave Eiffel. It stands 324 meters tall...

Hallucination Detection Results:
  🚨 x-vsr-hallucination-detected: true
  • x-vsr-hallucination-spans: ["originally painted red", "secret apartment"]
────────────────────────────────────────────────────────────
```
