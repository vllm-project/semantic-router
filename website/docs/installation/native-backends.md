---
title: Router Runtime
description: Run the models that classify requests, generate embeddings, and check safety.
---

Router Runtime runs the models used for routing: classifiers, embeddings, and
safety checks. The LLMs that answer users are configured separately in
[Model configuration](model-configuration.md).

## Choose a running mode

| | In-process models | External services |
| --- | --- | --- |
| Where the model runs | Inside the Router process | In a separate service |
| What you provide | Model files and a compatible CPU or GPU runtime | An API endpoint and credentials |
| Start here | [Run in-process models](runtime/in-process.md) | [Connect external services](runtime/external.md) |

Use in-process models to keep inference local. Use an external service when
you already serve the model elsewhere or want to manage its hardware separately.
Both modes can be used in the same Router.

## Configure a use case

- [Embeddings](runtime/embeddings.md): semantic matching, caches, and vector stores.
- [Safety models](runtime/safety.md): prompt guard, PII, and hallucination detection.

For startup failures, capacity limits, and configuration reloads, see
[Operations and troubleshooting](runtime/lifecycle-diagnostics.md).
