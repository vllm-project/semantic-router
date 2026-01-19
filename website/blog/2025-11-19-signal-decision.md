---
slug: signal-decision
title: "Signal-Decision Driven Architecture: Reshaping Semantic Routing at Scale"
authors: [Xunzhuo]
tags: [architecture, signal-decision, routing, vllm, semantic-router]
---

The earlier versions of vLLM Semantic Router relied on classification-based routing, a straightforward approach where user queries are classified into one of 14 MMLU domain categories, and then routed to corresponding models. While this worked for basic scenarios, we quickly discovered its limitations when building production AI systems for enterprises.

Synced from official vLLM Blog: [Signal-Decision Driven Architecture: Reshaping Semantic Routing at Scale](https://blog.vllm.ai/2025/11/19/signal-decision.html)

![banner](/img/signal-0.png)

---

<!-- truncate -->
