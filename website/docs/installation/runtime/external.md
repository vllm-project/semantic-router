---
title: External Inference
description: Connect named classification, guardrail, scoring, and embedding services.
---

# External inference

Use an external service when it owns the model process and hardware. The Router
owns the connector, request deadline, result validation, and admission budget.
The service receives the relevant request text or grounding inputs.

## Bind a named guardrail service

Merge this fragment into your canonical configuration and enable a jailbreak
signal/decision that consumes the guard. The address is an example: provide a
service implementing the documented classify contract.

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

The equivalent existing module setting is
`prompt_guard.backend: {protocol: http_classify, contract: label_distribution.v1, model: guard-service}`.
Use a recipe binding when different recipes need different deployments. The
module's `backend.deadline_ms` can restrict its call deadline.

## Wire protocol and result contract

| Task | Protocol / adapter | Contract and meaning |
| --- | --- | --- |
| Domain, generic sequence, prompt guard | `http_classify` | `label_distribution.v1`: complete declared-label distribution |
| PII | `http_classify` | `token_spans.v1`: scored entities with validated offsets |
| Complexity | `http_classify` | `score.v1` or a complete `label_distribution.v1` |
| Prompt guard | `http_chat` | `label_decision.v1`: categorical verdict, score unavailable when unreported |
| Hallucination detector | `http_chat` | `token_spans.v1` from context, question, and answer; chat-produced spans have no invented confidence |
| Generic LLM classifier | `http_chat` | `label_distribution.v1`: instructed, schema-constrained scored extraction |
| Text embedding | `openai_compatible` | `embedding.v1`: actual returned vector |

HTTP fact-check, feedback, modality, and NLI task adapters are not available.
Choosing a matching contract name alone does not enable them.

`http_classify` posts to the service's classify operation with
`{"inputs":"text"}`. It does **not** send `llm_model_name` as a model selector.
Two names pointing to the same operation URL and credentials call the same
service and share its physical admission identity. To select a different
server model, expose a distinct operation endpoint or use a protocol that
actually transmits a model selector.

Chat requests use the named model in the request and retain task-specific
prompting/parsing. A prompt-guard chat verdict is not the generic classifier's
scored JSON product. To use it, change the guard's adapter to `http_chat`, its
contract to `label_decision.v1`, and set `llm_model_name` on the external entry.
The service must implement the guard's supported response format.

Sequence distributions must contain the declared labels and valid scores;
subsets, duplicate/unknown labels, malformed JSON, and unsupported sigmoid
multi-label products are not converted to a probability distribution. Generic
LLM scores are model-reported confidence, not calibrated probabilities. See
[Classifier signals](../../tutorials/signal/learned/classifier.md).

## PII spans and grounding inputs

Remote PII entities supply a label, score, text, and start/end character
positions. Supported `byte_start`/`byte_end` metadata must agree with the text.
The adapter validates Unicode boundaries and produces UTF-8 byte offsets for
Router consumers. An outside label is not an entity. Missing score, unknown
label, invalid offsets, or an error envelope is a failed inference; a valid
empty entity list means no detected entities. Explicit partial/truncation
metadata is retained rather than silently treating a prefix as a complete scan.

Hallucination uses a separate structured request containing context, question,
and answer. It shares a span result contract with PII, not PII's raw-text
request semantics. Grounding spans remain relative to the answer. See
[Safety models](safety.md#grounding-and-nli).

## Time, capacity, and data

Connector timeouts and request cancellation bound the same operation budget,
including its admission wait. The Router validates response size, shape, and
semantics before using a result. It does not assume a remote classifier has
the local tokenizer or enforce a native `max_tokens` budget: configure the
external model's own limit and leave HTTP classifier deployment `input` unset.

Keep external credentials in the supported deployment secret/environment
mechanism and inspect only redacted diagnostics. Provider logging, retention,
and residency apply to the text sent to it. Embeddings use their own explicit
`api_key_env` option; see [Embeddings](embeddings.md#use-a-remote-text-provider).

## MCP classification

`global.model_catalog.modules.classifier.mcp` configures the existing MCP
classifier tool integration separately from model deployments. Its command or
remote transport, tool name, timeout, and response-size limit belong to that
module. The default maximum response body is 16 MiB. It is not a new HTTP
model-binding adapter.

The MCP classifier discovers categories and invokes the configured classify
tool. Follow its tool response contract for category, model recommendations,
and reasoning preference. Transport success alone does not supply a native
label distribution. See the
[MCP classifier configuration](../../tutorials/global/overview.md) and the
[configuration reference](../../api/configuration-schema.mdx).

## Migrate older remote guards

Canonical configuration rejects the retired module-level
`prompt_guard.protocol`. Run explicit migration:

```bash
vllm-sr config migrate --config config.yaml
```

Migration preserves a unique existing external guard name; a unique unnamed
entry receives `guardrail_classifier`. Multiple candidate guard entries or a
name conflict require an explicit selection. `http_classify` migrates to
`label_distribution.v1`; `http_chat` migrates to `label_decision.v1`. Review
the emitted configuration before serving. Runtime does not pick the first
external entry with a matching role.
