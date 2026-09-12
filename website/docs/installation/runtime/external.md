---
title: External services
description: Connect a separately hosted classifier, guard, or embedding service.
---

Use an external service when the model runs outside the Router. The Router
sends inputs to its API and uses the result in the configured routing policy.
The service manages the model and hardware.

## Connect a guard service

This example expects an HTTPS service that accepts `POST /classify` with
`{"inputs":"text"}` and returns scores for the guard's configured labels.
Merge it into your existing `config.yaml`, replacing the endpoint:

```yaml
global:
  model_catalog:
    external:
      - name: guard-service
        model_role: guardrail
        llm_endpoint:
          address: guard.example.com
          port: 443
          protocol: https
        llm_timeout_seconds: 5
        max_response_bytes: 1048576
    deployments:
      guard-http:
        provider: http
        external_model: guard-service
    modules:
      prompt_guard:
        enabled: true
        threshold: 0.7
        positive_labels: [INJECTION]
routing:
  model_bindings:
    prompt_guard:
      deployment: guard-http
      contract: label_distribution.v1
      adapter: http_classify
```

`guard-service` names the API connection; `guard-http` makes it available to
the recipe's prompt guard. To act on the result, configure a jailbreak rule
and decision as described in [Safety models](safety.md).

Run `vllm-sr config validate --config config.yaml`, then restart or reload the
Router. Configure credentials through the service's supported secret settings.
The service will receive the text it inspects.

## Choose a service type

| Service | Configuration | Used for |
| --- | --- | --- |
| Classify API | `adapter: http_classify` | Domain or custom classification, prompt guard, PII, complexity |
| Chat API | `adapter: http_chat` | Prompt guard, hallucination detection, LLM-based classification |
| Embedding API | `backend: openai_compatible` | [Remote text embeddings](embeddings.md#remote-embeddings) |
| MCP tool | `modules.classifier.mcp` | Classification through an existing MCP server |

The example uses a scored classifier. For a chat-based prompt guard, use
`contract: label_decision.v1`, change the adapter to `http_chat`, and set
`llm_model_name` on the external service. The service must return the supported
guard verdict format. See [Classifier signals](../../tutorials/signal/learned/classifier.md)
for generic classifier response formats.

The classify request contains only the input text; it does not send a model
name. Serve different classify models at different endpoints. Chat and
embedding requests include their configured model name.

## Service requirements

- Classification returns every configured label with a valid score. Missing,
  duplicate, or unknown labels cause an inference error.
- PII returns scored entities with valid text offsets. Hallucination detection
  instead receives context, question, and answer and returns answer-relative spans.
- Set request timeouts and response-size limits for the service. Leave local
  tokenizer `input` settings unset for HTTP classifiers; enforce token limits
  in the external service.
- External adapters are currently unavailable for fact-check, feedback,
  output-modality classification, and NLI.

For MCP, configure the transport, tool name, and timeout under
`global.model_catalog.modules.classifier.mcp`; see the
[configuration reference](../../api/configuration-schema.mdx).

Older configurations using `prompt_guard.protocol` should first run
`vllm-sr config migrate --config config.yaml` and select the intended external
service explicitly if more than one is configured.
