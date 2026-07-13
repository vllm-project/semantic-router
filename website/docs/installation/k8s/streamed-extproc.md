# Streamed ExtProc and immediate responses

This guide explains how to run vLLM Semantic Router behind an Envoy-compatible gateway when request bodies are delivered to ExtProc in streamed mode, and how streamed clients receive Semantic Router immediate responses such as looper, semantic-cache, and `fast_response` results.

Use this guide when you need one of the following:

- large OpenAI-compatible request bodies that should not be fully buffered by the gateway before ExtProc sees them;
- agentgateway `FullDuplexStreamed` ExtProc processing;
- Envoy AI Gateway or raw Envoy `STREAMED` request body processing;
- streamed Chat Completions clients (`"stream": true`) that may be short-circuited by Semantic Router before the upstream backend responds.

## How it works

Semantic Router is an Envoy External Processor. In buffered mode the gateway sends the full request body in one ExtProc message. In streamed mode the gateway sends multiple body chunks. Semantic Router's streamed body handler accumulates the chunks, applies the same routing and mutation pipeline at end-of-stream, and then emits one complete mutated request body or an immediate response.

For streamed Chat Completions responses, immediate responses keep OpenAI-compatible behavior:

- looper algorithms return `Content-Type: text/event-stream` when the original request has `"stream": true`;
- looper responses include `x-vsr-looper-*` headers such as `x-vsr-looper-model`, `x-vsr-looper-models-used`, `x-vsr-looper-iterations`, and `x-vsr-looper-algorithm`;
- non-streaming immediate responses, including many `fast_response` blocks, return a complete JSON response immediately;
- Response API requests are translated back through the Response API layer, so looper execution is forced to non-streaming internally for those requests.

:::note
"Streamed request body" and "streamed model response" are separate knobs. `request_body_mode: STREAMED` or `requestBodyMode: FullDuplexStreamed` controls how the gateway sends the request body to Semantic Router. The OpenAI request field `"stream": true` controls whether the client expects Server-Sent Events from the final model or immediate response.
:::

## Semantic Router configuration

Enable streamed request body handling in the Semantic Router runtime config. The setting lives under `global.router.streamed_body` in the canonical config.

```yaml
global:
  router:
    streamed_body:
      enabled: true
      max_bytes: 10485760   # reject larger accumulated bodies with 413
      timeout_sec: 30       # reject slow body accumulation with 408
```

Keep `max_bytes` high enough for your largest prompt or multimodal payload. Keep `timeout_sec` greater than the expected upload time between the first body chunk and end-of-stream.

The default reference config shows the same structure in `config/config.yaml`, and the streaming e2e profile uses it in `e2e/profiles/streaming/values.yaml`.

## Envoy AI Gateway / Envoy Gateway

For Envoy AI Gateway examples that use `EnvoyPatchPolicy`, change the Semantic Router ExtProc filter from buffered request bodies to streamed request bodies.

```yaml
apiVersion: gateway.envoyproxy.io/v1alpha1
kind: EnvoyPatchPolicy
metadata:
  name: ai-gateway-prepost-extproc-patch-policy
  namespace: default
spec:
  jsonPatches:
  - name: default/semantic-router/http
    operation:
      op: add
      path: /default_filter_chain/filters/0/typed_config/http_filters/0
      value:
        name: semantic-router-extproc
        typedConfig:
          '@type': type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
          failureModeAllow: false
          allowModeOverride: true
          grpcService:
            envoyGrpc:
              authority: semantic-router.vllm-semantic-router-system:50051
              clusterName: semantic-router
            timeout: 60s
          messageTimeout: 60s
          processingMode:
            requestHeaderMode: SEND
            requestBodyMode: STREAMED
            requestTrailerMode: SKIP
            responseHeaderMode: SEND
            responseBodyMode: BUFFERED
            responseTrailerMode: SKIP
```

The important fields are:

- `requestBodyMode: STREAMED` so request chunks are sent to ExtProc;
- `failureModeAllow: false` so an unavailable processor blocks the request
  instead of bypassing routing;
- `allowModeOverride: true` so Semantic Router can request per-route
  response-body processing changes when needed; and
- `messageTimeout` and `grpcService.timeout` large enough for classification and body accumulation.

A complete Kubernetes example is available in `deploy/kubernetes/streaming/aigw-resources/gwapi-resources.yaml`.

## agentgateway

agentgateway uses the Gateway API `AgentgatewayPolicy` abstraction rather than raw Envoy `processing_mode` names. For streamed bodies use `FullDuplexStreamed`.

```yaml
apiVersion: agentgateway.dev/v1alpha1
kind: AgentgatewayPolicy
metadata:
  name: semantic-router-extproc
  namespace: agentgateway-system
spec:
  targetRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: agentgateway-proxy
  traffic:
    # AgentgatewayPolicy ExtProc is fail-closed by default.
    extProc:
      backendRef:
        name: semantic-router
        namespace: agentgateway-system
        port: 50051
      processingOptions:
        requestHeaderMode: Send
        requestBodyMode: FullDuplexStreamed
        responseHeaderMode: Send
        responseBodyMode: Buffered
        allowModeOverride: true
```

The current Kubernetes `AgentgatewayPolicy` ExtProc schema does not expose a
failure-mode override. If the processor is unavailable, agentgateway returns
an error instead of bypassing semantic routing.

agentgateway does not support a separate `Streamed` request-body mode. Use `FullDuplexStreamed` for streamed request bodies and enable `global.router.streamed_body` in Semantic Router.

## Configure an immediate streamed looper response

Looper algorithms are the main immediate-response path added by the full-duplex streaming work. A decision with a looper algorithm and multiple `modelRefs` can return an immediate ExtProc response instead of forwarding the original request to one backend.

Example decision fragment:

```yaml
routing:
  decisions:
  - name: streamed_confidence_route
    priority: 100
    conditions:
      all:
      - signal: domain
        operator: equals
        value: code
    modelRefs:
    - modelName: small-code-model
      weight: 1
    - modelName: large-code-model
      weight: 1
    algorithm:
      type: confidence
      confidence:
        confidence_method: hybrid
        threshold: 0.72
        escalation_order: small_to_large
        on_error: skip
```

When the client sends `"stream": true`, Semantic Router calls the candidate model(s), aggregates the looper result, and returns an immediate SSE body to the gateway. The client still receives a normal OpenAI-compatible stream:

```bash
curl -N -i http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Write and explain a Python debounce decorator."}
    ],
    "max_tokens": 128
  }'
```

Look for:

- `HTTP/1.1 200 OK`;
- `content-type: text/event-stream`;
- `x-vsr-looper-algorithm: confidence`, `ratings`, or `remom`;
- SSE events beginning with `data: {"id":"chatcmpl-...","object":"chat.completion.chunk"...}`;
- final `data: [DONE]`.

## Configure a streamed-body safety block

`fast_response` can also short-circuit requests that arrive as streamed body chunks. This is useful for safety decisions such as PII or jailbreak blocking.

```yaml
routing:
  decisions:
  - name: streamed_jailbreak_block
    priority: 1000
    conditions:
      all:
      - signal: jailbreak
        operator: greater_than
        value: 0.6
    plugins:
    - type: fast_response
      configuration:
        message: This request was blocked by policy.
```

With `request_body_mode: STREAMED` or `requestBodyMode: FullDuplexStreamed`, Semantic Router accumulates the body, runs the safety signal at end-of-stream, and returns the configured immediate response without forwarding the request to the backend.

## Verification checklist

1. Confirm the gateway policy/filter is accepted:

   ```bash
   kubectl describe envoypatchpolicy ai-gateway-prepost-extproc-patch-policy -n default
   # or
   kubectl describe agentgatewaypolicy semantic-router-extproc -n agentgateway-system
   ```

2. Confirm Semantic Router has streamed body handling enabled:

   ```bash
   kubectl logs deploy/semantic-router -n vllm-semantic-router-system | grep -i streamed
   ```

3. Send a large or chunked request with `"model": "auto"` and verify it routes normally.

4. Send a streamed Chat Completions request with `"stream": true` that matches a looper decision and verify SSE output plus `x-vsr-looper-*` headers.

5. Send a request that matches a `fast_response` decision and verify the backend model is not called.

## Troubleshooting

- **Gateway accepts requests but Semantic Router never sees body chunks**: the ExtProc filter still uses buffered or skipped request body mode. Set Envoy `requestBodyMode: STREAMED` or agentgateway `requestBodyMode: FullDuplexStreamed`.
- **Request fails with 413**: the accumulated body exceeds `global.router.streamed_body.max_bytes`. Increase `max_bytes` or reduce request size.
- **Request fails with 408**: body chunks did not finish before `timeout_sec`. Increase `timeout_sec` or investigate client upload speed.
- **Client expected SSE but got JSON**: the OpenAI request did not include `"stream": true`, or the matched path is a non-streaming immediate response. Add `"stream": true` for Chat Completions looper routes and verify the matched decision.
- **agentgateway rejects `Streamed`**: agentgateway supports `FullDuplexStreamed`, not `Streamed`. Use `requestBodyMode: FullDuplexStreamed`.
- **Duplicate or partial upstream request body**: gateway and Semantic Router streamed modes are mismatched. Enable both the gateway streamed request-body mode and Semantic Router `streamed_body.enabled`.
