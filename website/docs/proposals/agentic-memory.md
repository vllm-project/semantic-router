---
title: Agentic Memory
description: Explores a proof-of-concept memory loop for retrieving, injecting, extracting, and storing user-scoped context.
created: 2026-02-09
status: Proof of concept
---

> **Status:** Proof of concept · **Created:** 2026-02-09

## Problem

Conversation history explains the current exchange, but it does not provide a durable,
selective memory across sessions. Sending all prior messages is expensive, may exceed
the context window, and increases privacy exposure. Saving every message as memory has
the opposite problem: irrelevant or incorrect details accumulate and can influence
future answers.

The useful product boundary is a small, user-scoped memory layer that retrieves only
relevant records and gives users a way to inspect or delete them.

## Proof of concept

The proposed loop is:

```mermaid
flowchart LR
  Request --> Identity["Resolve user identity"]
  Identity --> Retrieve["Retrieve relevant memory"]
  Retrieve --> Inject["Inject bounded memory context"]
  Inject --> Model
  Model --> Extract["Extract candidate memories"]
  Extract --> Validate["Validate and scope"]
  Validate --> Store["Persist or discard"]
```

Retrieval happens before generation. Candidate extraction happens after a successful
response. Either step may be disabled independently.

## Memory model

The proof of concept distinguishes three useful record types:

| Type | Purpose | Example |
| --- | --- | --- |
| Semantic | Stable facts or preferences about the user. | Preferred output language. |
| Procedural | Reusable instructions for how work should be done. | Use a specific review format. |
| Episodic | A bounded summary of a prior interaction. | The result of a completed troubleshooting session. |

Reflective memories, autonomous consolidation, and knowledge-graph inference are
outside the proof of concept. They require stronger provenance and correction rules.

Each stored record needs an owner identity, type, content, source reference, creation
time, and enough provenance to explain why it was saved. Embeddings are retrieval
indexes, not the authoritative record.

## Retrieval

Retrieval should:

1. require a stable user or tenant identity;
2. skip requests that clearly do not need memory;
3. form a search query from the current request and bounded recent context;
4. filter by owner before returning candidates;
5. apply top-k and similarity limits; and
6. inject memory as untrusted context, separate from system instructions.

A similarity score is not permission to disclose a record. Identity and policy filters
must run before content is exposed to the model.

## Saving and correction

The saving path should extract only durable facts, preferences, or procedures.
Transient requests, model speculation, secrets, and instructions copied from retrieved
content should not be saved automatically.

Users need operations to list and forget memories. Corrections should replace or
supersede an earlier record rather than rely on retrieval order to hide it. Every
write should retain its source so an operator can investigate contamination.

## Storage and failure behavior

The original proof of concept uses a vector store for semantic retrieval. That choice
does not establish a production storage contract. A production design still needs:

- indexed tenant isolation before vector search;
- retention and deletion guarantees;
- encryption and credential rotation;
- quotas and duplicate handling;
- backup and recovery; and
- behavior across multiple router replicas.

If retrieval is unavailable, the request should continue without memory when the route
allows it. Failed writes should be observable and must not change the model response.

## Scope and non-goals

This proof of concept validates the retrieve → inject → extract → store loop. It does
not claim production-scale isolation, reliable session-end detection, autonomous
reflection, or factual correctness of extracted memories.

The current [memory plugin guide](../tutorials/plugin/memory) is the source of truth
for implemented configuration and behavior. This proposal should not be used as a
deployment recipe.

## Evaluation

Evaluate retrieval and saving separately:

- retrieval relevance and cross-user isolation;
- rate of missed, irrelevant, and duplicate memories;
- extraction precision for durable facts and preferences;
- correction and deletion behavior;
- added latency and context tokens; and
- graceful degradation when identity or storage is unavailable.

Use synthetic tenants and inspectable records before testing with personal data.

## Open questions

- Which identity source is authoritative for each protocol?
- Should writes require explicit user consent or a route-level opt-in?
- How are conflicting memories resolved?
- Which record types need expiration by default?
- What isolation primitive is required before multi-tenant production use?

## References

- [Current memory plugin guide](../tutorials/plugin/memory)
- [Memory and replay in Router Learning](../tutorials/learning/memory-and-replay)
