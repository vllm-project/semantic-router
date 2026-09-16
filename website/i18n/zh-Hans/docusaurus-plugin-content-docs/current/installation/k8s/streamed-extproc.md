---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/k8s/streamed-extproc.md"
  outdated: false
---

# 流式 ExtProc 与立即响应

当请求体以流式模式送达 ExtProc，以及流式客户端接收 Semantic Router 立即响应（例如 looper、`response_cache` 和 `fast_response` 结果）时，本指南说明如何在 Envoy 兼容网关后面运行 vLLM Semantic Router。

在需要以下任一情况时使用本指南：

- 大型 OpenAI 兼容请求体，不应在 ExtProc 看到它们之前由网关完整缓冲；
- agentgateway `FullDuplexStreamed` ExtProc 处理；
- Envoy AI Gateway 或原始 Envoy `STREAMED` 请求体处理；
- 流式 Chat Completions 客户端（`"stream": true`），可能在上游后端响应之前被 Semantic Router 短路。

## 工作原理

Semantic Router 是 Envoy External Processor。在缓冲模式下，网关在一条 ExtProc 消息中发送完整请求体。在流式模式下，网关发送多个 body 分片。Semantic Router 的流式 body 处理程序累积这些分片，在流结束时应用相同的路由和改写流水线，然后发出一份完整的已改写请求体或立即响应。

对于流式 Chat Completions 响应，立即响应保持 OpenAI 兼容行为：

- 当原始请求带有 `"stream": true` 时，looper 算法返回 `Content-Type: text/event-stream`；
- looper 响应包含 `x-vsr-looper-*` 头，例如 `x-vsr-looper-model`、`x-vsr-looper-models-used`、`x-vsr-looper-iterations` 和 `x-vsr-looper-algorithm`；
- 非流式立即响应（包括许多 `fast_response` 块）立即返回完整 JSON 响应；
- Response API 请求会经 Response API 层回译，因此这些请求的 looper 执行在内部强制为非流式。

:::note
“流式请求体”和“流式模型响应”是分开的开关。`request_body_mode: STREAMED` 或 `requestBodyMode: FullDuplexStreamed` 控制网关如何将请求体发送给 Semantic Router。OpenAI 请求字段 `"stream": true` 控制客户端是否期望最终模型或立即响应返回 Server-Sent Events。
:::

## Semantic Router 配置

在 Semantic Router 运行时配置中启用流式请求体处理。该设置位于规范配置的 `global.router.streamed_body` 下。

```yaml
global:
  router:
    streamed_body:
      enabled: true
      max_bytes: 10485760   # reject larger accumulated bodies with 413
      timeout_sec: 30       # reject slow body accumulation with 408
```

将 `max_bytes` 保持得足够大，以容纳最大提示词或多模态载荷。将 `timeout_sec` 设为大于第一个 body 分片到流结束之间的预期上传时间。

上面的 10 MiB 和 30 秒值是与 `e2e/profiles/streaming/values.yaml` 中流式 e2e profile 匹配的示例护栏；它们不是运行时默认值，也不是经过实验校准的限制。省略任一值或将其设为 0 会禁用该护栏。参考 `config/config.yaml` 演示了更小的 1 MiB 和 15 秒策略。

## Envoy AI Gateway / Envoy Gateway

对于使用 `EnvoyPatchPolicy` 的 Envoy AI Gateway 示例，将 Semantic Router ExtProc 过滤器从缓冲请求体改为流式请求体。

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

重要字段是：

- `requestBodyMode: STREAMED`，以便将请求分片发送到 ExtProc；
- `allowModeOverride: true`，以便 Semantic Router 在需要时可以请求按路由更改响应体处理；
- `messageTimeout` 和 `grpcService.timeout` 足够大，以覆盖分类和 body 累积。

完整 Kubernetes 示例见 `deploy/kubernetes/streaming/aigw-resources/gwapi-resources.yaml`。

## agentgateway

agentgateway 使用 Gateway API `AgentgatewayPolicy` 抽象，而不是原始 Envoy `processing_mode` 名称。流式 body 使用 `FullDuplexStreamed`。

缓冲请求体在代理默认值和其他部署示例中仍然常见。附带的 agentgateway 示例在 `deploy/kubernetes/agentgateway/extproc-policy.yaml` 中显式选择流式；[agentgateway 安装指南](./agentgateway) 中的 Helm 命令显式启用 `global.router.streamed_body`。采用该示例时请同时使用这两项设置。

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

agentgateway 不支持单独的 `Streamed` 请求体模式。对流式请求体使用 `FullDuplexStreamed`，并在 Semantic Router 中启用 `global.router.streamed_body`。

Semantic Router 会检测协商后的 ExtProc body 模式。在 `FullDuplexStreamed` 下，它缓冲中间请求分片而不发出 body 替换，然后把完整处理后的请求作为一条流结束 `StreamedBodyResponse` 发送。在 Envoy `STREAMED` 下，它保留该模式要求的每分片一次响应行为。

## 配置立即流式 looper 响应

Looper 算法是全双工流式工作新增的主要立即响应路径。带有 looper 算法和多个 `modelRefs` 的决策可以返回立即 ExtProc 响应，而不是将原始请求转发到一个后端。

决策片段示例：

```yaml
routing:
  decisions:
    - name: streamed_confidence_route
      description: Escalate code requests when the first model is uncertain.
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: computer science
      modelRefs:
        - model: small-code-model
          use_reasoning: false
        - model: large-code-model
          use_reasoning: false
      algorithm:
        type: confidence
        confidence:
          confidence_method: hybrid
          threshold: 0.72
          escalation_order: small_to_large
          on_error: skip
```

`computer science` 信号和两个提供商模型也必须存在于同一配方中。完整约定见[置信度教程](/zh-Hans/docs/tutorials/algorithm/looper/confidence)。

当客户端发送 `"stream": true` 时，Semantic Router 调用候选模型，聚合 looper 结果，并向网关返回立即 SSE body。客户端仍收到正常的 OpenAI 兼容流：

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

查找：

- `HTTP/1.1 200 OK`；
- `content-type: text/event-stream`；
- `x-vsr-looper-algorithm: confidence`、`ratings` 或 `remom`；
- 以 `data: {"id":"chatcmpl-...","object":"chat.completion.chunk"...}` 开头的 SSE 事件；
- 最后的 `data: [DONE]`。

## 配置流式 body 安全拦截

`fast_response` 也可以短路以流式 body 分片到达的请求。这对 PII 或越狱拦截等安全决策很有用。

```yaml
routing:
  signals:
    jailbreak:
      - name: streamed_jailbreak
        method: classifier
        threshold: 0.6
        description: Detect prompt-injection attempts before forwarding.
  decisions:
    - name: streamed_jailbreak_block
      description: Return a policy response for detected prompt injection.
      priority: 1000
      rules:
        operator: AND
        conditions:
          - type: jailbreak
            name: streamed_jailbreak
      modelRefs: []
      plugins:
        - type: fast_response
          configuration:
            message: This request was blocked by policy.
```

在 `request_body_mode: STREAMED` 或 `requestBodyMode: FullDuplexStreamed` 下，Semantic Router 累积 body，在流结束时运行安全信号，并返回配置的立即响应，而不将请求转发到后端。

## 验证清单

1. 确认网关策略/过滤器已被接受：

 ```bash
   kubectl describe envoypatchpolicy ai-gateway-prepost-extproc-patch-policy -n default
   # or
   kubectl describe agentgatewaypolicy semantic-router-extproc -n agentgateway-system
   ```

2. 确认 Semantic Router 已启用流式 body 处理：

 ```bash
   kubectl logs deploy/semantic-router -n vllm-semantic-router-system | grep -i streamed
   ```

3. 发送带 `"model": "auto"` 的大型或分片请求，并验证其正常路由。

4. 发送匹配 looper 决策且带 `"stream": true` 的流式 Chat Completions 请求，并验证 SSE 输出以及 `x-vsr-looper-*` 头。

5. 发送匹配 `fast_response` 决策的请求，并验证未调用后端模型。

## 故障排查

- **网关接受请求，但 Semantic Router 从未看到 body 分片**：ExtProc 过滤器仍使用缓冲或跳过请求体模式。设置 Envoy `requestBodyMode: STREAMED` 或 agentgateway `requestBodyMode: FullDuplexStreamed`。
- **请求以 413 失败**：累积的 body 超过 `global.router.streamed_body.max_bytes`。增大 `max_bytes` 或减小请求大小。
- **请求以 408 失败**：body 分片未在 `timeout_sec` 之前完成。增大 `timeout_sec`，或检查客户端上传速度。
- **客户端期望 SSE 却收到 JSON**：OpenAI 请求未包含 `"stream": true`，或匹配路径是非流式立即响应。为 Chat Completions looper 路由添加 `"stream": true`，并验证匹配的决策。
- **agentgateway 拒绝 `Streamed`**：agentgateway 支持 `FullDuplexStreamed`，不支持 `Streamed`。使用 `requestBodyMode: FullDuplexStreamed`。
- **上游请求体重复或不完整**：网关与 Semantic Router 的流式模式不匹配。同时启用网关的流式请求体模式和 Semantic Router `streamed_body.enabled`。
