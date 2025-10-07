# Overview

Semantic Router's intelligent caching system understands the semantic meaning of queries, enabling cache hits for semantically similar requests and reducing LLM inference costs.

## Core Concepts

### Semantic Similarity

Uses embeddings and cosine similarity to match queries by meaning rather than exact text.

### Configurable Thresholds

Adjustable similarity thresholds balance cache hit rates with response quality.

### Multiple Backends

Support for in-memory, Redis, and Milvus backends for different scale requirements.

## How It Works

```mermaid
graph TB
    A[User Query] --> B[Generate Embedding]
    B --> C[Cache Lookup]
    C --> D{Similar Query?}

    D -->|Hit| E[Return Cached Response]
    D -->|Miss| F[Route to LLM]

    F --> G[Store Response in Cache]
    G --> H[Return Response]

    style E fill:#90EE90
    style G fill:#FFB6C1
```

## Backend Options

### In-Memory Cache

Fast, local caching for development and single-instance deployments.

### Milvus Cache

Persistent, distributed caching using vector database for production environments.

## Key Benefits

- **Cost Reduction**: Avoid redundant LLM API calls for similar queries
- **Improved Latency**: Cache hits return responses in milliseconds
- **Better Throughput**: Handle more concurrent requests efficiently
- **Semantic Understanding**: Match queries by meaning, not just text
