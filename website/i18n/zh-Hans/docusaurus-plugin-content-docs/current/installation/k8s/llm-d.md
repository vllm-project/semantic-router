---
title: 与 llm-d 集成
description: 由 Semantic Router 选择模型池，由 llm-d 在该池内调度副本。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/installation/k8s/llm-d.md"
  outdated: false
---

# 与 llm-d 集成

当请求必须做两种不同选择时，使用此拓扑：

1. **Semantic Router** 根据请求意图、策略和配方状态选择逻辑模型或模型池。
2. **llm-d** 使用负载、前缀缓存或其他端点信息，在该池内选择健康副本。

不要配置两套系统做同一决策。Semantic Router 不应选择 Pod，llm-d 也不应决定请求适用哪项业务策略或模型家族。

```text
client
  -> inference gateway
  -> Semantic Router ExtProc
  -> HTTPRoute selected by x-selected-model
  -> InferencePool
  -> llm-d endpoint picker
  -> model replica
```

## 开始之前

加入 Semantic Router 之前，先独立部署并验证 llm-d。llm-d 的发布产物、CRD、chart 和插件配置一起演进，因此使用一个受支持的 llm-d 版本，不要混用从不同版本复制的清单。

- 遵循当前 [llm-d quickstart](https://llm-d.ai/docs/getting-started/quickstart) 或合适的 well-lit path。
- 选择受支持的[网关集成](https://llm-d.ai/docs/infrastructure/gateway)。
- 使用 [llm-d artifacts 参考](https://llm-d.ai/docs/api-reference/artifacts) 获取匹配的 Gateway API Inference Extension 资源和 chart。

在此边界上，你应已能在没有 Semantic Router 的情况下，通过网关向每个 `InferencePool` 发送请求。

## 集成约定

在两套系统中保持这些名称对齐：

| 名称 | 所有者 | 要求 |
| --- | --- | --- |
| 提供商模型 | Semantic Router | `providers.models[].name` 是配方所选的逻辑池名称。 |
| 请求头 | Semantic Router | Router 将所选提供商模型写入 `x-selected-model`。 |
| 路由匹配 | Gateway API | 一条 `HTTPRoute` 精确匹配该头值。 |
| 后端引用 | Gateway API / llm-d | 路由指向预期的 `InferencePool`。 |
| 已服务模型 | 模型服务器 | 池中的副本接受网关转发的模型标识。 |

例如，此路由将 Router 的 `local/code` 选择映射到现有的 llm-d 池。按部署调整名称和命名空间：

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: code-pool
  namespace: inference
spec:
  parentRefs:
    - name: llm-d-inference-gateway
  rules:
    - matches:
        - headers:
            - type: Exact
              name: x-selected-model
              value: local/code
      backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: code-pool
          port: 8000
```

该路由不配置池或 endpoint picker。这些仍属于 llm-d 部署，并且必须使用该版本支持的 API 版本。

## 加入 Semantic Router

1. 创建完整的规范 Router 配置，其提供商模型名称与路由值匹配。部署前先校验：

 ```bash
   vllm-sr config validate --config config.yaml
   ```

2. 按[配置工作流](../configuration-workflows) 中的 Helm 或 Operator 工作流部署 Semantic Router。直接使用 Helm 时，使用 `configOverride`，以便原子替换 chart 示例配置。

3. 将 Semantic Router 作为 ExtProc 服务接入网关。确切资源因网关而异；受支持的接入模式见 [Gateway API Inference Extension](gateway-api-inference-extension)。

4. 在规范 Router 配置中保持 `global.router.clear_route_cache: true`。网关必须在下游请求路径中调用 ExtProc，保留其清除路由缓存的响应，然后在 Semantic Router 写入 `x-selected-model` 后重新评估 `HTTPRoute`。

## 逐层验证

先检查资源状态，不要依赖生成的 Pod 名称：

```bash
kubectl get gateway,httproute -A
kubectl get inferencepools -A
kubectl get httproute code-pool -n inference \
  -o jsonpath='{.status.parents[*].conditions[?(@.type=="ResolvedRefs")].status}{"\n"}'
```

然后使用当前 Router 配置暴露的虚拟模型发送请求：

```bash
curl -i "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Review this function for a race condition."}]
  }'
```

验证全部三项决策，不要在 HTTP 200 处停止：

- 响应包含预期的 `x-vsr-selected-model` 值；
- 匹配的 `HTTPRoute` 报告 `ResolvedRefs=True`；以及
- llm-d 从预期的 `InferencePool` 选择了就绪端点。

## 常见故障

| 现象 | 检查 |
| --- | --- |
| 路由从未匹配 | 比较 `x-selected-model` 与 `HTTPRoute` 值，包括大小写和命名空间。 |
| `ResolvedRefs=False` | 检查 `InferencePool` 名称、group、端口和跨命名空间权限。 |
| 池正确但已服务模型错误 | 将提供商的模型标识与副本接受的模型名称对齐。 |
| Semantic Router 被绕过 | 确认网关在路由匹配之前调用 ExtProc。 |
| EPP 没有端点 | 诊断 llm-d 池选择器、Pod 就绪状态，以及与发行版匹配的插件配置。 |

## 生产边界

- 固定 Semantic Router、llm-d、网关、CRD 和模型服务器镜像。
- 不要把提供商凭证放在 Router YAML 中；在拥有凭证的组件上使用基于 Secret 的 binding。
- 明确失败行为。fail-open ExtProc 策略可能绕过语义策略；fail-closed 策略在 Router 不可用时会停止全部流量。
- 在组合上线之前，分别测试直接池访问、语义选择和端点调度。
