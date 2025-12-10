# Router Memory

This guide shows you how to use Router Memory with the OpenAI Response API to enable stateful conversations with conversation chaining, session persistence, and multi-turn context management.

## Overview

Router Memory enables the semantic-router to store and retrieve conversation history, allowing LLM agents to maintain context across multiple requests. It implements the [OpenAI Response API](https://platform.openai.com/docs/api-reference/responses) specification, providing:

- **Conversation Chaining**: Link responses via `previous_response_id` for multi-turn conversations
- **Session Persistence**: Store responses with configurable TTL
- **Pluggable Backends**: Memory (default), Milvus, and Redis storage options
- **Automatic Translation**: Seamless translation between Response API and Chat Completions API

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Request                              │
│                    POST /v1/responses                                │
│         { "input": "...", "previous_response_id": "resp_xxx" }      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Semantic Router (extproc)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Request         │  │ Response Store  │  │ Response            │  │
│  │ Translator      │  │ (Memory/Milvus/ │  │ Translator          │  │
│  │                 │  │  Redis)         │  │                     │  │
│  │ Response API    │  │                 │  │ Chat Completions    │  │
│  │ → Chat Compl.   │  │ Conversation    │  │ → Response API      │  │
│  └─────────────────┘  │ History         │  └─────────────────────┘  │
│                       └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Backend LLM (vLLM)                           │
│                    POST /v1/chat/completions                         │
│               { "messages": [...conversation history...] }           │
└─────────────────────────────────────────────────────────────────────┘
```

## Supported Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/responses` | POST | Create a new response |
| `/v1/responses/{id}` | GET | Retrieve a stored response |
| `/v1/responses/{id}` | DELETE | Delete a stored response |
| `/v1/responses/{id}/input_items` | GET | List input items for a response |

## Configuration

### Basic Configuration

```yaml
# config.yaml
response_api:
  enabled: true
  store_backend: "memory"  # Options: memory, milvus, redis
  ttl_seconds: 86400       # 24 hours (default: 30 days)
  max_responses: 1000      # Maximum stored responses
```

### Memory Backend (Default)

Best for development, testing, and single-instance deployments.

```yaml
response_api:
  enabled: true
  store_backend: "memory"
  ttl_seconds: 86400
  memory:
    max_responses: 10000
    max_conversations: 5000
```

### Milvus Backend (Coming Soon)

Best for production with vector search capabilities.

```yaml
response_api:
  enabled: true
  store_backend: "milvus"
  ttl_seconds: 2592000  # 30 days
  milvus:
    address: "localhost:19530"
    database: "semantic_router"
    response_collection: "responses"
    conversation_collection: "conversations"
```

### Redis Backend (Coming Soon)

Best for distributed deployments with multiple router instances.

```yaml
response_api:
  enabled: true
  store_backend: "redis"
  ttl_seconds: 2592000
  redis:
    config_path: "/path/to/redis-config.yaml"
```

## Usage Examples

### 1. Create Initial Response

```bash
curl -X POST http://localhost:8801/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "input": "What is the capital of France?",
    "instructions": "You are a helpful geography expert."
  }'
```

Response:

```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1733834567,
  "model": "qwen3",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [{"type": "text", "text": "The capital of France is Paris."}]
    }
  ]
}
```

### 2. Continue Conversation (Chaining)

Use the returned `id` as `previous_response_id` for follow-up questions:

```bash
curl -X POST http://localhost:8801/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "input": "What is its population?",
    "previous_response_id": "resp_abc123"
  }'
```

The router automatically:

1. Retrieves the conversation chain from storage
2. Builds the full message history
3. Sends to backend with complete context

### 3. Retrieve Response History

```bash
curl http://localhost:8801/v1/responses/resp_abc123
```

### 4. List Input Items

```bash
curl http://localhost:8801/v1/responses/resp_abc123/input_items
```

### 5. Delete Response

```bash
curl -X DELETE http://localhost:8801/v1/responses/resp_abc123
```

## Full Configuration Example

See [config.response-api.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/testing/config.response-api.yaml):

```yaml
bert_model:
  model_id: models/all-MiniLM-L12-v2
  threshold: 0.6
  use_cpu: true

# Response API Configuration
response_api:
  enabled: true
  store_backend: "memory"
  ttl_seconds: 86400
  max_responses: 1000

# vLLM Endpoints
vllm_endpoints:
  - name: "primary"
    address: "vllm-server"
    port: 8000
    weight: 1

model_config:
  "qwen3":
    use_reasoning: false
    preferred_endpoints: ["primary"]

default_model: "qwen3"
```

## Key Features

### Conversation Chaining

The `previous_response_id` field enables multi-turn conversations:

```text
Response 1 (id: resp_001)
    └── Response 2 (id: resp_002, previous_response_id: resp_001)
            └── Response 3 (id: resp_003, previous_response_id: resp_002)
```

When a request includes `previous_response_id`, the router:

1. Fetches the entire conversation chain from storage
2. Reconstructs the message history in chronological order
3. Appends the new user input
4. Sends the complete context to the backend LLM

### Automatic API Translation

| Response API Field | Chat Completions Equivalent |
|--------------------|----------------------------|
| `input` (string) | `messages[].content` (role: user) |
| `instructions` | `messages[0]` (role: system) |
| `previous_response_id` | Expanded to full `messages` array |
| `max_output_tokens` | `max_tokens` |
| `temperature` | `temperature` |

### Storage Backends Comparison

| Feature | Memory | Milvus | Redis |
|---------|--------|--------|-------|
| Persistence | ❌ | ✅ | ✅ |
| Distributed | ❌ | ✅ | ✅ |
| Vector Search | ❌ | ✅ | ❌ |
| Native TTL | ✅ | ❌ | ✅ |
| Setup Complexity | Low | Medium | Low |
| Best For | Dev/Test | Semantic Search | Production |

## Response Headers

The router adds custom headers to responses:

| Header | Description |
|--------|-------------|
| `x-vsr-selected-model` | The model used for the response |
| `x-vsr-selected-reasoning` | Whether reasoning was enabled |
| `x-vsr-injected-system-prompt` | Whether a system prompt was injected |

## Error Handling

| Error | HTTP Code | Description |
|-------|-----------|-------------|
| Response not found | 404 | The `previous_response_id` or response ID doesn't exist |
| Invalid request | 400 | Missing required `input` field |
| Storage error | 500 | Backend storage failure |

## Roadmap

- [ ] **Milvus Backend** ([#803](https://github.com/vllm-project/semantic-router/issues/803)) - Persistent storage with vector search
- [ ] **Redis Backend** ([#804](https://github.com/vllm-project/semantic-router/issues/804)) - Distributed caching
- [ ] **Context Engineering** ([#806](https://github.com/vllm-project/semantic-router/issues/806)) - Smart context compression
- [ ] **Semantic Session Affinity** ([#807](https://github.com/vllm-project/semantic-router/issues/807)) - KV cache-aware routing
- [ ] **Agentic Memory** ([#808](https://github.com/vllm-project/semantic-router/issues/808)) - Persistent structured memory

## Reference

- [OpenAI Response API Documentation](https://platform.openai.com/docs/api-reference/responses)
- [PR #802: Response API Implementation](https://github.com/vllm-project/semantic-router/pull/802)
