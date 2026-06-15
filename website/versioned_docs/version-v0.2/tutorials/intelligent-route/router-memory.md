# Multi Turn Conversations

Router Memory enables stateful conversations via the [OpenAI Response API](https://platform.openai.com/docs/api-reference/responses), supporting conversation chaining with `previous_response_id`.

## Overview

Semantic Router acts as the **unified brain** for multiple LLM backends that only support the Chat Completions API. It provides:

- **Cross-Model Stateful Conversations**: Maintain conversation history across different models
- **Unified Response API**: Single API interface regardless of backend model
- **Transparent Translation**: Automatic conversion between Response API and Chat Completions

```mermaid
flowchart TB
    subgraph Clients
        C1[Agent A]
        C2[Agent B]
    end

    subgraph SR["Semantic Router"]
        API[Response API]
        Store[(Conversation<br/>Store)]
    end

    subgraph Backends["LLM Backends (Chat Completions Only)"]
        M1[GPT-4o]
        M2[Claude]
        M3[Qwen]
        M4[Llama]
    end

    C1 & C2 --> API
    API <--> Store
    API --> M1 & M2 & M3 & M4
```

With Router Memory, you can start a conversation with one model and continue it with another—the conversation history is preserved in the router, not in any single backend.

## Request Flow

```mermaid
flowchart TB
    subgraph Client
        A1[POST /v1/responses]
    end

    subgraph Router["Semantic Router (extproc)"]
        B1[Receive Request]
        B2{Has previous_response_id?}
        B3[Load Conversation Chain]
        B4[Translate to Chat Completions]
        B5[Translate to Response API]
        B6[Store Response]
    end

    subgraph Store["Response Store"]
        C1[(Memory)]
    end

    subgraph Backend["Backend LLM"]
        D1[POST /v1/chat/completions]
    end

    A1 --> B1
    B1 --> B2
    B2 -->|Yes| B3
    B2 -->|No| B4
    B3 --> C1
    C1 --> B4
    B4 --> D1
    D1 --> B5
    B5 --> B6
    B6 --> C1
    B6 --> A1
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/responses` | POST | Create a new response |
| `/v1/responses/{id}` | GET | Retrieve a stored response |
| `/v1/responses/{id}` | DELETE | Delete a stored response |
| `/v1/responses/{id}/input_items` | GET | List input items for a response |

## Configuration

```yaml
response_api:
  enabled: true
  store_backend: "memory"   # Currently only "memory" is supported
  ttl_seconds: 86400        # Default: 30 days
  max_responses: 1000
```

## Usage

### 1. Create Response

```bash
curl -X POST http://localhost:8801/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "input": "Tell me a joke.",
    "instructions": "Remember my name is Xunzhuo. Then I will ask you!",
    "temperature": 0.7,
    "max_output_tokens": 100
  }'
```

Response:

```json
{
  "id": "resp_7cb437001e1ad5b84b6dd8ef",
  "object": "response",
  "status": "completed",
  "output": [{
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "Sure thing, Xunzhuo! Why don't scientists trust atoms? Because they make up everything! 😄"}]
  }],
  "usage": {"input_tokens": 94, "output_tokens": 75, "total_tokens": 169}
}
```

### 2. Continue Conversation

Use `previous_response_id` to chain conversations:

```bash
curl -X POST http://localhost:8801/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "input": "What is my name?",
    "previous_response_id": "resp_7cb437001e1ad5b84b6dd8ef",
    "max_output_tokens": 100
  }'
```

Response:

```json
{
  "id": "resp_ec2822df62e390dcb87aa61d",
  "status": "completed",
  "output": [{
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "Your name is Xunzhuo."}]
  }],
  "previous_response_id": "resp_7cb437001e1ad5b84b6dd8ef"
}
```

### 3. Get Response

```bash
curl http://localhost:8801/v1/responses/resp_7cb437001e1ad5b84b6dd8ef
```

### 4. List Input Items

```bash
curl http://localhost:8801/v1/responses/resp_7cb437001e1ad5b84b6dd8ef/input_items
```

Response:

```json
{
  "object": "list",
  "data": [{
    "type": "message",
    "role": "system",
    "content": [{"type": "input_text", "text": "Remember my name is Xunzhuo."}]
  }],
  "has_more": false
}
```

### 5. Delete Response

```bash
curl -X DELETE http://localhost:8801/v1/responses/resp_7cb437001e1ad5b84b6dd8ef
```

## API Translation

| Response API | Chat Completions |
|--------------|------------------|
| `input` | `messages[].content` (role: user) |
| `instructions` | `messages[0]` (role: system) |
| `previous_response_id` | Expanded to full `messages` array |
| `max_output_tokens` | `max_tokens` |

## Reference

- [OpenAI Response API](https://platform.openai.com/docs/api-reference/responses)
