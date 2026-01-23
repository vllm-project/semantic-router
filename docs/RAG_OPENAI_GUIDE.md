# OpenAI RAG Integration Guide

This guide demonstrates how to use OpenAI's File Store and Vector Store APIs for RAG (Retrieval-Augmented Generation) in Semantic Router, following the [OpenAI Responses API cookbook](https://cookbook.openai.com/examples/rag_on_pdfs_using_file_search).

## Overview

The OpenAI RAG backend integrates with OpenAI's File Store and Vector Store APIs to provide a first-class RAG experience. It supports two workflow modes:

1. **Direct Search Mode** (default): Synchronous retrieval using vector store search API
2. **Tool-Based Mode**: Adds `file_search` tool to request (Responses API workflow)

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│     Semantic Router                 │
│  ┌───────────────────────────────┐  │
│  │      RAG Plugin               │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │  OpenAI RAG Backend     │  │  │
│  │  └──────┬──────────────────┘  │  │
│  └─────────┼──────────────────── ┘  │
└────────────┼─────────────────────── ┘
             │
             ▼
┌─────────────────────────────────────┐
│      OpenAI API                     │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ File Store   │  │Vector Store  │ │
│  │   API        │  │   API        │ │
│  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────┘
```

## Prerequisites

1. OpenAI API key with access to File Store and Vector Store APIs
2. Files uploaded to OpenAI File Store
3. Vector store created and populated with files

## Configuration

### Basic Configuration

Add the OpenAI RAG backend to your decision configuration:

```yaml
decisions:
  - name: rag-openai-decision
    signals:
      - type: keyword
        keywords: ["research", "document", "knowledge"]
    plugins:
      rag:
        enabled: true
        backend: "openai"
        backend_config:
          vector_store_id: "vs_abc123"  # Your vector store ID
          api_key: "${OPENAI_API_KEY}"  # Or use environment variable
          max_num_results: 10
          workflow_mode: "direct_search"  # or "tool_based"
```

### Advanced Configuration

```yaml
rag:
  enabled: true
  backend: "openai"
  similarity_threshold: 0.7
  top_k: 10
  max_context_length: 5000
  injection_mode: "tool_role"  # or "system_prompt"
  on_failure: "skip"  # or "warn" or "block"
  cache_results: true
  cache_ttl_seconds: 3600
  backend_config:
    vector_store_id: "vs_abc123"
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"  # Optional, defaults to OpenAI
    max_num_results: 10
    file_ids:  # Optional: restrict search to specific files
      - "file-123"
      - "file-456"
    filter:  # Optional: metadata filter
      category: "research"
      published_date: "2024-01-01"
    workflow_mode: "direct_search"  # or "tool_based"
    timeout_seconds: 30
```

## Workflow Modes

### 1. Direct Search Mode (Default)

Synchronous retrieval using vector store search API. Context is retrieved before sending the request to the LLM.

**Use Case**: When you need immediate context injection and want to control the retrieval process.

**Example**:

```yaml
backend_config:
  workflow_mode: "direct_search"
  vector_store_id: "vs_abc123"
```

**Flow**:

1. User sends query
2. RAG plugin calls vector store search API
3. Retrieved context is injected into request
4. Request sent to LLM with context

### 2. Tool-Based Mode (Responses API)

Adds `file_search` tool to the request. The LLM calls the tool automatically, and results appear in response annotations.

**Use Case**: When using Responses API and want the LLM to control when to search.

**Example**:

```yaml
backend_config:
  workflow_mode: "tool_based"
  vector_store_id: "vs_abc123"
```

**Flow**:

1. User sends query
2. RAG plugin adds `file_search` tool to request
3. Request sent to LLM
4. LLM calls `file_search` tool
5. Results appear in response annotations

## Usage Examples

### Example 1: Basic RAG Query

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-VSR-Selected-Decision: rag-openai-decision" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "What is Deep Research?"
      }
    ]
  }'
```

### Example 2: Responses API with file_search Tool

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "input": "What is Deep Research?",
    "tools": [
      {
        "type": "file_search",
        "file_search": {
          "vector_store_ids": ["vs_abc123"],
          "max_num_results": 5
        }
      }
    ]
  }'
```

### Example 3: Python Client

```python
import requests

# Direct search mode
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "X-VSR-Selected-Decision": "rag-openai-decision"
    },
    json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is Deep Research?"}
        ]
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

## File Store Operations

The OpenAI RAG backend includes a File Store client for managing files:

### Upload File

```go
import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/openai"

client := openai.NewFileStoreClient("https://api.openai.com/v1", apiKey)
file, err := client.UploadFile(ctx, fileReader, "document.pdf", "assistants")
```

### Create Vector Store

```go
vectorStoreClient := openai.NewVectorStoreClient("https://api.openai.com/v1", apiKey)
store, err := vectorStoreClient.CreateVectorStore(ctx, &openai.CreateVectorStoreRequest{
    Name:    "my-vector-store",
    FileIDs: []string{"file-123", "file-456"},
})
```

### Attach File to Vector Store

```go
_, err := vectorStoreClient.CreateVectorStoreFile(ctx, "vs_abc123", "file-123")
```

## Testing

### Unit Tests

Run unit tests for OpenAI RAG:

```bash
cd src/semantic-router
go test ./pkg/openai/... -v
go test ./pkg/extproc/req_filter_rag_openai_test.go -v
```

### E2E Tests

Run E2E tests based on the OpenAI cookbook:

```bash
# Python-based E2E test
python e2e/testing/08-rag-openai-test.py --base-url http://localhost:8080

# Go-based E2E test (requires Kubernetes cluster)
make e2e-test E2E_TESTS=rag-openai
```

## Monitoring and Observability

The OpenAI RAG backend exposes the following metrics:

- `rag_retrieval_attempts_total{backend="openai", decision="...", status="success|error"}`
- `rag_retrieval_latency_seconds{backend="openai", decision="..."}`
- `rag_similarity_score{backend="openai", decision="..."}`
- `rag_context_length_chars{backend="openai", decision="..."}`
- `rag_cache_hits_total{backend="openai"}`
- `rag_cache_misses_total{backend="openai"}`

### Tracing

OpenTelemetry spans are created for:

- `semantic_router.rag.retrieval` - RAG retrieval operation
- `semantic_router.rag.context_injection` - Context injection operation

## Error Handling

The RAG plugin supports three failure modes:

- **skip** (default): Continue without context, log warning
- **warn**: Continue with warning header
- **block**: Return error response (503)

```yaml
rag:
  on_failure: "skip"  # or "warn" or "block"
```

## Best Practices

1. **Use Direct Search for Synchronous Workflows**: When you need immediate context injection
2. **Use Tool-Based for Responses API**: When using Responses API and want LLM-controlled search
3. **Cache Results**: Enable caching for frequently accessed queries
4. **Set Appropriate Timeouts**: Configure `timeout_seconds` based on your vector store size
5. **Filter Results**: Use `file_ids` or `filter` to narrow search scope
6. **Monitor Metrics**: Track retrieval latency and similarity scores

## Troubleshooting

### No Results Found

- Verify vector store ID is correct
- Check that files are attached to the vector store
- Ensure files have completed processing (check `file_counts.completed`)

### High Latency

- Reduce `max_num_results`
- Enable result caching
- Use `file_ids` to limit search scope

### Authentication Errors

- Verify API key is correct
- Check API key has access to File Store and Vector Store APIs
- Ensure base URL is correct (if using custom endpoint)

## References

- [OpenAI Responses API Cookbook - RAG on PDFs](https://cookbook.openai.com/examples/rag_on_pdfs_using_file_search)
- [OpenAI File Store API Documentation](https://platform.openai.com/docs/api-reference/files)
- [OpenAI Vector Store API Documentation](https://platform.openai.com/docs/api-reference/vector-stores)
