---
translation:
  source_commit: "9052d81a"
  source_file: "docs/installation/k8s/streamed-extproc.md"
  outdated: false
---

# 流式 ExtProc 与即时响应

本指南介绍了当请求体以流式模式传递给 ExtProc 时，如何在兼容 Envoy 的网关后运行 vLLM Semantic Router，以及流式客户端如何接收 Semantic Router 的即时响应，例如 looper、semantic cache 和 `fast_response` 结果。

在以下场景中，请使用本指南：

- 大型 OpenAI 兼容请求体，不应在 ExtProc 收到之前由网关完整缓冲；
- agentgateway 的 `FullDuplexStreamed` ExtProc 处理；
- Envoy AI Gateway 或原生 Envoy 的 `STREAMED` 请求体处理；
- 可能在上游后端响应前被 Semantic Router 短路的流式 Chat Completions 客户端（`"stream": true`）。

## 工作原理

Semantic Router 是一个 Envoy External Processor。在缓冲模式下，网关通过一条 ExtProc 消息发送完整请求体。在流式模式下，网关会发送多个请求体分块。Semantic Router 的流式请求体处理器会累积分块，在流结束时应用同一套路由和修改流水线，然后发出一个完整的已修改请求体或即时响应。

对于流式 Chat Completions 响应，即时响应会保持 OpenAI 兼容行为：

- 当原始请求包含 `"stream": true` 时，looper 算法返回 `Content-Type: text/event-stream`；
- looper 响应包含 `x-vsr-looper-*` header，例如 `x-vsr-looper-model`、`x-vsr-looper-models-used`、`x-vsr-looper-iterations` 和 `x-vsr-looper-algorithm`；
- 非流式即时响应（包括许多 `fast_response` 拦截）会立即返回完整的 JSON 响应；
- Responses API 请求会通过 Responses API 层转换回来，因此这些请求在内部会被强制使用非流式 looper 执行。

:::note
“流式请求体”和“流式模型响应”是两个独立开关。`request_body_mode: STREAMED` 或 `requestBodyMode: FullDuplexStreamed` 控制网关如何将请求体发送给 Semantic Router；OpenAI 请求字段 `"stream": true` 则控制客户端是否期望从最终模型或即时响应中接收 Server-Sent Events。
:::

## Semantic Router 配置

在 Semantic Router 运行时配置中启用流式请求体处理。该设置位于规范配置的 `global.router.streamed_body` 下。

```yaml
global:
  router:
    streamed_body:
      enabled: true
      max_bytes: 10485760   # 累积请求体超过此大小时返回 413
      timeout_sec: 30       # 请求体累积过慢时返回 408
```

请确保 `max_bytes` 足以容纳最大的 prompt 或多模态 payload。`timeout_sec` 应大于从第一个请求体分块到流结束之间的预期上传时间。

上例中的 10 MiB 和 30 秒是与 `e2e/profiles/streaming/values.yaml` 中流式 E2E profile 一致的示例保护值，并非运行时默认值，也不是经过实验校准的限制。省略任一值或将其设置为零都会禁用对应保护。参考配置 `config/config.yaml` 展示了更小的 1 MiB 和 15 秒策略。

## Envoy AI Gateway / Envoy Gateway

对于使用 `EnvoyPatchPolicy` 的 Envoy AI Gateway 示例，请将 Semantic Router ExtProc 过滤器从缓冲请求体改为流式请求体。

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

关键字段包括：

- `requestBodyMode: STREAMED`，将请求分块发送给 ExtProc；
- `allowModeOverride: true`，允许 Semantic Router 在需要时请求按路由更改响应体处理方式；
- 足以覆盖分类和请求体累积时间的 `messageTimeout` 与 `grpcService.timeout`。

完整的 Kubernetes 示例位于 `deploy/kubernetes/streaming/aigw-resources/gwapi-resources.yaml`。

## agentgateway

agentgateway 使用 Gateway API 的 `AgentgatewayPolicy` 抽象，而不是原生 Envoy `processing_mode` 名称。对于流式请求体，请使用 `FullDuplexStreamed`。

缓冲请求体仍是代理默认配置和其他部署示例中的常见选择。项目自带的 agentgateway 示例在 `deploy/kubernetes/agentgateway/extproc-policy.yaml` 中显式启用流式处理；[agentgateway 安装指南](./agentgateway)中的 Helm 命令则显式启用 `global.router.streamed_body`。采用该示例时，请同时使用这两项设置。

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
        requestTrailerMode: Send
        responseTrailerMode: Send
        allowModeOverride: true
```

agentgateway 不支持单独的 `Streamed` 请求体模式。流式请求体应使用 `FullDuplexStreamed`，并在 Semantic Router 中启用 `global.router.streamed_body`。

Semantic Router 会检测协商后的 ExtProc 请求体模式。使用 `FullDuplexStreamed` 时，它会缓冲中间请求分块而不发出请求体替换，并在流结束时通过一条 `StreamedBodyResponse` 发送完整的已处理请求。使用 Envoy `STREAMED` 时，则保留该模式所要求的每分块一条响应行为。

## 配置 looper 的流式即时响应

Looper 算法是全双工流式处理所新增的主要即时响应路径。包含 looper 算法和多个 `modelRefs` 的 decision 可以直接返回 ExtProc 即时响应，而不必将原始请求转发给单个后端。

Decision 配置片段示例：

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

当客户端发送 `"stream": true` 时，Semantic Router 会调用候选模型、聚合 looper 结果，并向网关返回即时 SSE body。客户端仍会收到正常的 OpenAI 兼容 stream：

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

请检查：

- `HTTP/1.1 200 OK`；
- `content-type: text/event-stream`；
- `x-vsr-looper-algorithm: confidence`、`ratings` 或 `remom`；
- 以 `data: {"id":"chatcmpl-...","object":"chat.completion.chunk"...}` 开头的 SSE event；
- 最后的 `data: [DONE]`。

## 配置流式请求体的安全拦截

`fast_response` 也可以短路以流式请求体分块到达的请求。这适用于 PII 或 jailbreak 拦截等安全 decision。

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

使用 `request_body_mode: STREAMED` 或 `requestBodyMode: FullDuplexStreamed` 时，Semantic Router 会累积请求体，在流结束时运行安全信号，并直接返回配置的即时响应，而不会将请求转发给后端。

## 验证清单

1. 确认网关 policy/filter 已被接受：

   ```bash
   kubectl describe envoypatchpolicy ai-gateway-prepost-extproc-patch-policy -n default
   # 或
   kubectl describe agentgatewaypolicy semantic-router-extproc -n agentgateway-system
   ```

2. 确认 Semantic Router 已启用流式请求体处理：

   ```bash
   kubectl logs deploy/semantic-router -n vllm-semantic-router-system | grep -i streamed
   ```

3. 使用 `"model": "auto"` 发送大型或分块请求，并验证其正常路由。

4. 发送匹配 looper decision 且包含 `"stream": true` 的流式 Chat Completions 请求，并验证 SSE 输出和 `x-vsr-looper-*` header。

5. 发送匹配 `fast_response` decision 的请求，并验证后端模型未被调用。

## 故障排除

- **网关接受请求，但 Semantic Router 从未收到请求体分块**：ExtProc 过滤器仍使用缓冲或跳过请求体模式。请设置 Envoy `requestBodyMode: STREAMED` 或 agentgateway `requestBodyMode: FullDuplexStreamed`。
- **请求返回 413**：累积请求体超过 `global.router.streamed_body.max_bytes`。请增大 `max_bytes` 或缩小请求体。
- **请求返回 408**：请求体分块未能在 `timeout_sec` 前完成。请增大 `timeout_sec` 或检查客户端上传速度。
- **客户端期望 SSE，但收到 JSON**：OpenAI 请求未包含 `"stream": true`，或匹配的是非流式即时响应路径。对于 Chat Completions looper 路由，请添加 `"stream": true` 并验证匹配的 decision。
- **agentgateway 拒绝 `Streamed`**：agentgateway 支持的是 `FullDuplexStreamed`，而不是 `Streamed`。请使用 `requestBodyMode: FullDuplexStreamed`。
- **上游请求体重复或不完整**：网关与 Semantic Router 的流式模式不匹配。请同时启用网关流式请求体模式和 Semantic Router 的 `streamed_body.enabled`。
