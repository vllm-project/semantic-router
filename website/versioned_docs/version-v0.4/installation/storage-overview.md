---
title: Data and Storage
description: Choose backing services for cache, memory, replay, response history, and vector data.
---

# Data and Storage

Semantic Router can run without an external vector database, but several
optional capabilities persist or share data. Choose stores by capability,
durability, scale, and data-handling policy rather than by product name alone.

## What may be stored

| Capability | Typical data | Why it is stored |
| --- | --- | --- |
| Response cache | Request representation and prior response | Reuse a previous answer for an equivalent request. |
| Agentic memory | Embedded memories and metadata | Retrieve relevant information across turns or sessions. |
| Vector store | Documents, chunks, embeddings, and file metadata | Support retrieval and uploaded knowledge. |
| Router Replay | Route metadata and, when enabled, bounded request and response bodies | Inspect and evaluate routing behavior. |
| Response API | Recent response records | Retrieve generated responses through the management surface. |

These stores have different privacy and retention implications. A local model
route is not end-to-end private if its prompts or responses are written to an
unapproved shared store.

## Available guides

### Valkey

[Valkey Agentic Memory](valkey-memory) describes a Redis-compatible,
single-service option with vector search. It is useful for smaller deployments
or teams that already operate Valkey.

### Qdrant

[Qdrant](qdrant) covers Docker and Kubernetes deployment plus Router bindings.
It can back semantic cache, memory, vector-store, and replay configurations.

### Milvus

[Milvus](milvus) covers persistent vector storage, Kubernetes deployment,
monitoring, migration, and recovery. It is suited to larger or shared vector
collections, but has more operational dependencies than a single-service
store.

### In-memory stores

In-memory backends are useful for local experiments and short-lived caches.
They do not provide restart durability or cross-replica sharing. Do not infer
that ephemeral means insensitive: content remains readable within the running
process and may still appear in logs or diagnostics.

## Selection questions

Before enabling a store, decide:

1. Which Router capability will use it?
2. Must data survive a Router restart?
3. Must several Router replicas share the same state?
4. What request or response content can be captured?
5. What are the retention, deletion, encryption, and access requirements?
6. Who backs up and restores the service?

Then configure the shared store under `global.stores` or the relevant service
block, and enable only the route plugins that need it. See
[Stores and Tools](../tutorials/global/stores-and-tools) for the canonical
configuration surface.
