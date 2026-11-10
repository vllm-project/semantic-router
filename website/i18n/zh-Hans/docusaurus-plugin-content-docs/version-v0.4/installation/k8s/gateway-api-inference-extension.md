---
title: Gateway API 推理扩展
description: 将语义模型池选择与 Kubernetes InferencePool 内的端点选择结合。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/installation/k8s/gateway-api-inference-extension.md"
  outdated: false
---

# Gateway API 推理扩展

Gateway API Inference Extension（GAIE）和 Semantic Router 解决路由的不同部分：

- **Semantic Router** 根据请求含义、策略和配方状态选择逻辑模型或池。
- **GAIE** 将该池表示为 `InferencePool`，并让 endpoint picker 选择就绪副本。

当一个公开模型 ID 可以在多个模型池之间选择，且每个池包含多个可互换的服务副本时，将两者一起使用。若每个模型直接映射到一个 Kubernetes Service，请改用更简单的网关集成，例如 [Istio](istio)。

```text
client
  -> Gateway
  -> Semantic Router ExtProc
  -> HTTPRoute chosen by x-selected-model
  -> InferencePool
  -> endpoint picker
  -> model replica
```

## 选择并安装网关栈

使用平台支持的网关实现安装并验证 GAIE。使用上游项目的同一兼容集合；不要混用从不同版本复制的 CRD、chart 和示例。

- [GAIE 文档](https://gateway-api-inference-extension.sigs.k8s.io/)
- [支持的网关实现](https://gateway-api-inference-extension.sigs.k8s.io/implementations/gateways/)
- [GAIE 发行版](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases)
- [llm-d 网关提供方](https://llm-d.ai/docs/infrastructure/gateway)

加入 Semantic Router 之前，确认：

1. `Gateway` 报告 `Programmed=True`；
2. 每个 `HTTPRoute` 报告已接受且引用已解析；
3. 每个 `InferencePool` 有就绪端点；以及
4. 直接请求能到达预期的池。

Semantic Router 不安装、也不拥有网关控制器、GAIE CRD、endpoint picker 或模型服务器。

## 定义名称约定

Semantic Router 所选的提供商模型会成为 Gateway API 路由使用的请求头。该精确值必须在以下三个对象中一致：

```yaml
# Router config fragment
providers:
  defaults:
    model: local/general
  models:
    - name: local/general
      provider_model_id: served-general
      api_format: openai
      backend_refs:
        - name: general-pool
          endpoint: general-pool.inference.svc.cluster.local:8000
          protocol: http
          provider: vllm
          weight: 100
```

```yaml
# Gateway API fragment
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: general-pool
  namespace: inference
spec:
  parentRefs:
    - name: inference-gateway
  rules:
    - matches:
        - headers:
            - type: Exact
              name: x-selected-model
              value: local/general
      backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: general-pool
          port: 8000
```

Router 在请求上写入 `x-selected-model`。它在响应上以 `x-vsr-selected-model` 向客户端暴露逻辑选择。GAIE 负责随后在 `general-pool` 内选择端点。

## 部署 Semantic Router

应用前先创建并校验完整配置：

```bash
vllm-sr config validate --config config.yaml
```

然后按 [Helm 或 Operator 工作流](../configuration-workflows) 部署。直接使用 Helm 时，通过 `configOverride` 传入完整的规范文档；这会在 chart 应用其 Kubernetes 侧改写之前替换 chart 示例配置。

## ExtProc 之后重新评估路由

网关必须在下游请求路径中调用 Semantic Router，并在写入 `x-selected-model` 后重新评估路由。在规范配置中保持此 Router 设置启用：

```yaml
global:
  router:
    clear_route_cache: true
```

这要求 Envoy 丢弃 ExtProc 之前选定的路由，并再次评估 `HTTPRoute` 头匹配。确保网关的 ExtProc 策略保留该响应标志。仅将依赖路由的授权和其他过滤器放在重新评估之后的路由上，或明确验证其顺序。

- **Istio：** 使用 ExtProc `EnvoyFilter` 及其服务 `DestinationRule`。[Istio 指南](istio) 展示了直接 Service 版本的接入方式。
- **agentgateway：** 在其预路由阶段附加 `AgentgatewayPolicy`。见 [agentgateway](agentgateway)。
- **Envoy AI Gateway / Envoy Gateway：** 使用网关支持的 ExtProc 策略面。见 [Envoy AI Gateway](ai-gateway)。

不要把一种网关实现的接入资源应用到另一种；它们的策略 API 和处理模式不可互换。

对于分片请求体或立即流式响应，还需按 [Streamed ExtProc](streamed-extproc) 配置 body 模式。

## 验证组合路径

使用当前配置暴露的虚拟模型发送请求：

```bash
curl -i "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Explain this API error."}]
  }'
```

逐层检查：

```bash
kubectl get gateway,httproute -A
kubectl get inferencepools -A
kubectl get httproute general-pool -n inference \
  -o jsonpath='{.status.parents[*].conditions[*].type}{" "}{.status.parents[*].conditions[*].status}{"\n"}'
```

响应的 `x-vsr-selected-model` 应匹配路由的 `x-selected-model` 值，且 endpoint picker 应报告该池中的就绪端点。仅有 HTTP 200 并不能证明语义选择和端点选择都已运行。

## 按所有权排查

| 现象 | 从这里开始 |
| --- | --- |
| 响应没有 `x-vsr-selected-model` 头 | Semantic Router 配方选择和 ExtProc 接入 |
| 头存在但路由未被选中 | `HTTPRoute` 头值、路由状态和网关处理顺序 |
| `ResolvedRefs=False` | `InferencePool` 名称、group、端口、命名空间和引用权限 |
| 池已选中但没有后端响应 | Endpoint-picker 状态、池选择器和模型服务器就绪状态 |
| 仅流式或大型请求失败 | ExtProc 请求体模式、body 限制和超时 |

## 生产清单

- 固定经过测试的 Gateway API、GAIE、网关、endpoint-picker、Router 和模型服务器版本。
- 为每条路由决定 ExtProc 失败是 fail-open 还是 fail-closed。
- 将提供商凭证和网关 TLS 材料放在各自所属的 Secret 工作流中，不要放在 Router YAML 里。
- 在启用组合路径之前，将直接池访问、语义池选择和端点调度作为分开的故障域测试。
