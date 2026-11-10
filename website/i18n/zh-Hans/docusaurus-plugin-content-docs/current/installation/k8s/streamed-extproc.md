---
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/installation/k8s/streamed-extproc.md"
  outdated: false
---

# 流式 ExtProc 与立即响应

本指南说明当请求体以流式模式送达 ExtProc 时，如何在 Envoy 兼容网关后面运行 vLLM Semantic Router，以及流式客户端如何接收 Semantic Router 的立即响应，例如执行器、`response_cache` 和 `fast_response` 的结果。

在需要以下任一情况时使用本指南：

- 大型 OpenAI 兼容请求体，不应在 ExtProc 看到它们之前由网关完整缓冲；
- agentgateway `FullDuplexStreamed` ExtProc 处理；
- Agent Router（原 Envoy AI Gateway）或原始 Envoy 的 `STREAMED` 请求体处理，或原始 Envoy 的 `FULL_DUPLEX_STREAMED`；
- 流式 Chat Completions 客户端（`"stream": true`），可能在上游后端响应之前被 Semantic Router 短路。

## 工作原理 {#how-it-works}

Semantic Router 是 Envoy External Processor：

- **缓冲模式。**网关在一条 ExtProc 消息中发送完整请求体。
- **流式模式。**网关发送多个 body 分片。Semantic Router 的流式 body 处理程序累积这些分片，在流结束时应用相同的路由和改写流水线，然后发出一份完整的已改写请求体或立即响应。

指定具体模型的请求与 `vllm-sr/auto` 请求以相同方式累积，`streamed_body.max_bytes` 和 `streamed_body.timeout_sec` 限制同样适用于它们。它们的分片会保留到流结束，因为流水线仍可能把模型改写为提供方的模型 ID、把请求翻译为后端的 API 格式，或为流式 Chat Completions 请求添加 `stream_options.include_usage`。

每个分片在复制进请求体缓冲区之前都会检查剩余的 `max_bytes` 预算；请求体恰好达到限额时允许通过。如果大小或超时守卫拒绝请求，Semantic Router 只发送一次立即响应（413 或 408），释放请求体 handler 并结束 ExtProc 流。后续分片或 trailers 不会恢复路由处理，也不会调用请求体插件。

对于流式 Chat Completions 响应，立即响应保持 OpenAI 兼容行为：

- 当原始请求带有 `"stream": true` 时，执行器（`escalate`、`fusion`、`remom`、`ratings` 和 `workflows`）返回 `Content-Type: text/event-stream`；
- 执行器的响应在 `x-vsr-trace` 的 `looper` 成员中报告调用的模型、算法和迭代次数；
- 非流式立即响应（包括许多 `fast_response` 块）立即返回完整 JSON 响应；
- Response API 请求会经 Response API 层回译，因此对这些请求，执行器在内部以非流式运行。

:::note
“流式请求体”和“流式模型响应”是分开的开关。`request_body_mode: STREAMED` 或 `requestBodyMode: FullDuplexStreamed` 控制网关如何将请求体发送给 Semantic Router。OpenAI 请求字段 `"stream": true` 控制客户端是否期望最终模型或立即响应返回 Server-Sent Events。
:::

## Semantic Router 配置 {#semantic-router-configuration}

在 Semantic Router 运行时配置中启用流式请求体处理。该设置位于规范配置的 `global.router.streamed_body` 下。

```yaml
global:
  router:
    streamed_body:
      enabled: true
      max_bytes: 10485760   # reject larger accumulated bodies with 413
      timeout_sec: 30       # reject slow body accumulation with 408
```

将 `max_bytes` 保持得足够大，以容纳最大的提示词或多模态载荷。将 `timeout_sec` 设为大于第一个 body 分片到流结束之间的预期上传时间。

Router 记录三个按 `recipe` 标注的序列，用于确定这些守卫的大小：

| 指标 | 测量 | 用途 |
| --- | --- | --- |
| `llm_streamed_body_arrival_seconds` | 上传时间，从第一个 body 分片到流结束 | `timeout_sec`：设为高于代表性时间窗口内的 p99 或最大值 |
| `llm_streamed_body_bytes` | 累积的请求体大小 | `max_bytes` |
| `llm_streamed_body_chunks` | 网关发送的 body 消息数 | 了解网关如何切分请求体 |

例如，一小时内上传时间的 p99 为
`histogram_quantile(0.99, sum by (le) (rate(llm_streamed_body_arrival_seconds_bucket[1h])))`。这些序列只在 STREAMED 和 FULL_DUPLEX_STREAMED 模式下、请求解析到入口之后记录；指定具体后端模型的请求使用 `recipe="unknown"`。

:::note
上面的 10 MiB 和 30 秒是与 `e2e/profiles/streaming/values.yaml` 中流式 e2e profile 匹配的示例护栏，不是运行时默认值，也不是经过校准的限制。省略任一值或将其设为 0 会禁用该护栏。参考 `config/config.yaml` 使用更小的 1 MiB 和 15 秒策略。
:::

使用 Kubernetes Operator 时，在 `spec.config.streamed_body` 下设置相同的字段；Operator 会把它们渲染到 `global.router.streamed_body`：

```yaml
spec:
  gateway:
    existingRef:
      name: shared-gateway
      namespace: gateway-system
  config:
    streamed_body:
      enabled: true
      max_bytes: 10485760
      timeout_sec: 30
```

Operator 的 standalone 模式直接提供 HTTP，没有 Envoy sidecar。`spec.config.streamed_body` 设置在已有 Gateway 以流式模式调用 ExtProc 时生效。原始 Envoy 部署必须按下文所述选择自己的 body 处理模式。

## Agent Router / Envoy Gateway

对于使用 `EnvoyPatchPolicy` 的 Agent Router 示例，将 Semantic Router ExtProc 过滤器从缓冲请求体改为流式请求体。

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

## 原始 Envoy {#raw-envoy}

如果原始 Envoy 的路由表或后端依赖 Semantic Router 设置的请求头（`vllm-sr serve` 生成的配置就是如此），请使用 `request_body_mode: BUFFERED` 或 `FULL_DUPLEX_STREAMED`，不要使用 `STREAMED`：

- **为什么不用 `STREAMED`。**在 `STREAMED` 模式下，Semantic Router 一应答请求头，Envoy 就把它们继续传下去。在该应答之后到达的请求体仍可被改写，但 Envoy 不再对该请求应用[请求头修改](https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto#envoy-v3-api-field-service-ext-proc-v3-commonresponse-header-mutation)。
- **会出什么问题。**Semantic Router 在流结束时设置提供方凭据、提供方请求路径和 `x-vsr-route-model` 路由头，因此这样的请求会带着原始路径和客户端自己的 `Authorization` 请求头到达 Envoy 的默认路由。请求是否受影响取决于其请求体多久才到达，因此故障是间歇性的。
- **可行的配置。**在 `request_body_mode: FULL_DUPLEX_STREAMED`、`request_trailer_mode: SEND` 且启用 `global.router.streamed_body` 时，Envoy 会应用这些请求头修改，因为 Semantic Router 会把对请求头的应答保留到请求体完成路由。

:::caution
请让该过滤器的 `failure_mode_allow` 保持默认值 `false`。在应答被保留期间，失败的 ExtProc 流否则会让 Envoy 带着客户端原始的请求头转发请求。
:::

上面的 Agent Router 示例基于 `x-ai-eg-model` 路由，Semantic Router 不设置该请求头，因此它的路由不依赖这些请求头修改。

## agentgateway {#agentgateway}

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

要基于 `x-vsr-route-model` 或 Semantic Router 设置的其他请求头路由，请在 `traffic` 下添加 `phase: PreRouting`：在默认的 `PostRouting` 阶段，agentgateway 会在 ExtProc 运行之前选择路由。

Semantic Router 会检测协商后的 ExtProc body 模式：

- **在 `FullDuplexStreamed` 下**，它缓冲中间请求分片而不发出 body 替换，并把对请求头的应答保留到请求体完成路由。然后它先发送带有路由请求头修改的请求头应答，再把完整处理后的请求作为一条 `StreamedBodyResponse` 发送。请求带有 trailers 时，trailers 标志请求体结束，并在最后应答。
- **在 Envoy `STREAMED` 下**，它保留该模式要求的每分片一次响应行为。

## 配置立即流式执行器响应 {#configure-an-immediate-streamed-executor-response}

执行器是主要的立即响应路径。算法为执行器、并带有多个 `modelRefs` 的决策可以返回立即 ExtProc 响应，而不是将原始请求转发到一个后端。

决策片段示例：

```yaml
recipes:
  - name: default
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
          type: escalate
          escalate:
            order: size
            accept:
              confidence: {method: hybrid, gte: 0.72}
```

`computer science` 信号和两个提供商模型也必须存在于同一配方中。完整约定见 [Escalate 教程](../../tutorials/algorithm/looper/escalate.md)。

当客户端发送 `"stream": true` 时，Semantic Router 调用候选模型，聚合执行器的结果，并向网关返回立即 SSE body。客户端仍收到正常的 OpenAI 兼容流：

```bash
curl -N -i http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
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
- `x-vsr-response-path: looper`；
- 以 `data: {"id":"chatcmpl-...","object":"chat.completion.chunk"...}` 开头的 SSE 事件；
- 最后的 `data: [DONE]`。

## 配置流式 body 安全拦截 {#configure-a-streamed-body-safety-block}

`fast_response` 也可以短路以流式 body 分片到达的请求。这对 PII 或越狱拦截等安全决策很有用。

```yaml
recipes:
  - name: default
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

## 验证清单 {#verification-checklist}

1. 确认网关策略或过滤器已被接受：

   ```bash
   kubectl describe envoypatchpolicy ai-gateway-prepost-extproc-patch-policy -n default
   # or
   kubectl describe agentgatewaypolicy semantic-router-extproc -n agentgateway-system
   ```

2. 确认 Semantic Router 已启用流式 body 处理：

   ```bash
   kubectl logs deploy/semantic-router -n vllm-semantic-router-system | grep -i streamed
   ```

3. 发送带 `"model": "vllm-sr/auto"` 的大型或分片请求，并验证其正常路由。

4. 发送匹配执行器决策且带 `"stream": true` 的流式 Chat Completions 请求，验证 SSE 输出；对 `global.router.headers.debug` 允许的调用方，还要验证 `x-vsr-trace` 的 `looper` 成员。

5. 发送匹配 `fast_response` 决策的请求，并验证未调用后端模型。

## 故障排查 {#troubleshooting}

- **网关接受请求，但 Semantic Router 从未看到 body 分片**：ExtProc 过滤器仍使用缓冲或跳过请求体模式。设置 Envoy `requestBodyMode: STREAMED` 或 agentgateway `requestBodyMode: FullDuplexStreamed`。
- **请求以 413 失败**：累积的 body 超过 `global.router.streamed_body.max_bytes`。把 `max_bytes` 与 `llm_streamed_body_bytes` 对比，然后增大 `max_bytes` 或减小请求大小。
- **请求以 408 失败**：body 分片未在 `timeout_sec` 之前完成。把 `timeout_sec` 与 `llm_streamed_body_arrival_seconds` 的高分位数对比，然后增大 `timeout_sec`，或检查客户端上传速度。
- **客户端期望 SSE 却收到 JSON**：OpenAI 请求未包含 `"stream": true`，或匹配路径是非流式立即响应。为 Chat Completions 执行器路由添加 `"stream": true`，并验证匹配的决策。
- **agentgateway 拒绝 `Streamed`**：agentgateway 支持 `FullDuplexStreamed`，不支持 `Streamed`。使用 `requestBodyMode: FullDuplexStreamed`。
- **上游请求体重复或不完整**：网关与 Semantic Router 的流式模式不匹配。同时启用网关的流式请求体模式和 Semantic Router `streamed_body.enabled`。
- **部分请求带着客户端的 `Authorization` 请求头到达默认后端**：某个原始 Envoy 过滤器使用了 `request_body_mode: STREAMED`。请改为 `BUFFERED`，或改为 `FULL_DUPLEX_STREAMED` 并设置 `request_trailer_mode: SEND`、启用 `streamed_body`，见[原始 Envoy](#raw-envoy)。
