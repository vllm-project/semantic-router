---
translation:
  source_commit: "0907152"
  source_file: "docs/tutorials/observability/distributed-tracing.md"
  outdated: false
---

# 使用 OpenTelemetry 进行分布式追踪 (Distributed Tracing)

本指南介绍如何在 vLLM Semantic Router 中配置和使用分布式追踪，以增强可观测性和调试能力。

## 概览

vLLM Semantic Router 使用 OpenTelemetry 实现了全面的分布式追踪，提供对请求处理管道的细粒度可见性。追踪可以帮助您：

- **调试生产问题**：跟踪单个请求在整个路由管道中的路径
- **优化性能**：识别分类、缓存和路由中的瓶颈
- **监控安全**：跟踪 PII 检测和越狱防御操作
- **分析决策**：了解路由逻辑和推理模式 (Reasoning Mode) 的选择
- **关联服务**：连接路由和 vLLM 后端之间的追踪信息

## 架构

### 追踪层级 (Trace Hierarchy)

一个典型的请求追踪遵循以下结构：

```
semantic_router.request.received [根 span]
├─ semantic_router.classification (分类)
├─ semantic_router.security.pii_detection (PII 检测)
├─ semantic_router.security.jailbreak_detection (越狱检测)
├─ semantic_router.cache.lookup (缓存查找)
├─ semantic_router.routing.decision (路由决策)
├─ semantic_router.backend.selection (后端选择)
├─ semantic_router.system_prompt.injection (系统提示词注入)
└─ semantic_router.upstream.request (上游请求)
```

### Span 属性

每个 span 都包含丰富的属性，遵循 LLM 可观测性的 OpenInference 规范：

**请求元数据：**

- `request.id` - 唯一请求标识符
- `user.id` - 用户标识符（如果可用）
- `http.method` - HTTP 方法
- `http.path` - 请求路径

**模型信息：**

- `model.name` - 选定的模型名称
- `routing.original_model` - 原始请求的模型
- `routing.selected_model` - 路由选择的模型

**分类：**

- `category.name` - 分类结果类别
- `classifier.type` - 分类器实现类型
- `classification.time_ms` - 分类耗时

**安全：**

- `pii.detected` - 是否发现 PII
- `pii.types` - 检测到的 PII 类型
- `jailbreak.detected` - 是否检测到越狱尝试
- `security.action` - 采取的操作（拦截、允许）

**路由：**

- `routing.strategy` - 路由策略（自动、指定）
- `routing.reason` - 路由决策原因
- `reasoning.enabled` - 是否启用推理模式
- `reasoning.effort` - 推理努力等级

**性能：**

- `cache.hit` - 缓存命中/未命中状态
- `cache.lookup_time_ms` - 缓存查找耗时
- `processing.time_ms` - 总处理时间

## 配置

### 基础配置

在您的 `config.yaml` 中添加 `observability.tracing` 部分：

```yaml
observability:
  tracing:
    enabled: true
    provider: "opentelemetry"
    exporter:
      type: "stdout"  # 或 "otlp"
      endpoint: "localhost:4317"
      insecure: true
    sampling:
      type: "always_on"  # 或 "probabilistic"
      rate: 1.0
    resource:
      service_name: "vllm-semantic-router"
      service_version: "v0.1.0"
      deployment_environment: "production"
```

### 配置选项

#### Exporter 类型

**stdout** - 将追踪打印到控制台（开发环境）

```yaml
exporter:
  type: "stdout"
```

**otlp** - 导出到兼容 OTLP 的后端（生产环境）

```yaml
exporter:
  type: "otlp"
  endpoint: "jaeger:4317"  # Jaeger, Tempo, Datadog 等
  insecure: true  # 生产环境中配合 TLS 使用 false
```

#### 采样策略 (Sampling Strategies)

**always_on** - 对所有请求进行采样（开发/调试）

```yaml
sampling:
  type: "always_on"
```

**always_off** - 禁用采样（紧急性能处理）

```yaml
sampling:
  type: "always_off"
```

**probabilistic** - 按百分比对请求进行采样（生产环境）

```yaml
sampling:
  type: "probabilistic"
  rate: 0.1  # 采样 10% 的请求
```

### 环境特定配置

#### 开发环境 (Development)

```yaml
observability:
  tracing:
    enabled: true
    provider: "opentelemetry"
    exporter:
      type: "stdout"
    sampling:
      type: "always_on"
    resource:
      service_name: "vllm-semantic-router-dev"
      deployment_environment: "development"
```

#### 生产环境 (Production)

```yaml
observability:
  tracing:
    enabled: true
    provider: "opentelemetry"
    exporter:
      type: "otlp"
      endpoint: "tempo:4317"
      insecure: false  # 使用 TLS
    sampling:
      type: "probabilistic"
      rate: 0.1  # 10% 采样率
    resource:
      service_name: "vllm-semantic-router"
      service_version: "v0.1.0"
      deployment_environment: "production"
```

## 部署

### 配合 Jaeger

1. **启动 Jaeger**（用于测试的一体化版本）：

```bash
docker run -d --name jaeger \
  -p 4317:4317 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

1. **配置路由**：

```yaml
observability:
  tracing:
    enabled: true
    exporter:
      type: "otlp"
      endpoint: "localhost:4317"
      insecure: true
    sampling:
      type: "probabilistic"
      rate: 0.1
```

1. **访问 Jaeger UI**：http://localhost:16686

### 配合 Grafana Tempo

1. **配置 Tempo** (tempo.yaml)：

```yaml
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317

storage:
  trace:
    backend: local
    local:
      path: /tmp/tempo/traces
```

1. **启动 Tempo**：

```bash
docker run -d --name tempo \
  -p 4317:4317 \
  -p 3200:3200 \
  -v $(pwd)/tempo.yaml:/etc/tempo.yaml \
  grafana/tempo:latest \
  -config.file=/etc/tempo.yaml
```

1. **配置路由**：

```yaml
observability:
  tracing:
    enabled: true
    exporter:
      type: "otlp"
      endpoint: "tempo:4317"
      insecure: true
```

### Kubernetes 部署

```yaml
apiversion: v1
kind: ConfigMap
metadata:
  name: router-config
data:
  config.yaml: |
    observability:
      tracing:
        enabled: true
        exporter:
          type: "otlp"
          endpoint: "jaeger-collector.observability.svc:4317"
          insecure: false
        sampling:
          type: "probabilistic"
          rate: 0.1
        resource:
          service_name: "vllm-semantic-router"
          deployment_environment: "production"
---
apiversion: apps/v1
kind: Deployment
metadata:
  name: semantic-router
spec:
  template:
    spec:
      containers:
      - name: router
        image: vllm-semantic-router:latest
        env:
        - name: CONFIG_PATH
          value: /config/config.yaml
        volumeMounts:
        - name: config
          mountPath: /config
      volumes:
      - name: config
        configMap:
          name: router-config
```

## 使用示例

### 查看追踪

#### 控制台输出 (stdout exporter)

```json
{
  "Name": "semantic_router.classification",
  "SpanContext": {
    "TraceID": "abc123...",
    "SpanID": "def456..."
  },
  "Attributes": [
    {
      "Key": "category.name",
      "Value": "math"
    },
    {
      "Key": "classification.time_ms",
      "Value": 45
    }
  ],
  "Duration": 45000000
}
```

#### Jaeger UI

1. 导航至 http://localhost:16686
2. 选择服务：`vllm-semantic-router`
3. 点击 "Find Traces"
4. 查看追踪详情和时间线

### 性能分析

**查找慢请求：**

```
Service: vllm-semantic-router
Min Duration: 1s
Limit: 20
```

**分析分类瓶颈：**
按操作过滤：`semantic_router.classification`
按耗时排序（降序）

**跟踪缓存有效性：**
按标签过滤：`cache.hit = true`
对比命中与未命中的耗时

### 调试问题

**查找失败请求：**
按标签过滤：`error = true`

**追踪特定请求：**
按标签过滤：`request.id = req-abc-123`

**查找 PII 违规：**
按标签过滤：`security.action = blocked`

## 追踪上下文传播 (Trace Context Propagation)

路由使用 W3C Trace Context Header 自动传播追踪上下文：

**请求 Header**（由路由提取）：

```
traceparent: 00-abc123-def456-01
tracestate: vendor=value
```

**上游 Header**（由路由注入）：

```
traceparent: 00-abc123-ghi789-01
x-vsr-destination-endpoint: endpoint1
x-selected-model: gpt-4
```

这实现了从 客户端 → 路由 → vLLM 后端 的端到端追踪。

## 性能注意事项

### 开销 (Overhead)

配置得当时，追踪带来的开销极小：

- **全量采样 (Always-on)**：延迟增加约 1-2%
- **10% 概率采样**：延迟增加约 0.1-0.2%
- **异步导出**：Span 导出不会阻塞请求处理

### 优化建议

1. **生产环境使用概率采样**

    ```yaml
    sampling:
      type: "probabilistic"
      rate: 0.1  # 根据流量进行调整
    ```

2. **动态调整采样率**
    - 高流量：0.01-0.1 (1-10%)
    - 中等流量：0.1-0.5 (10-50%)
    - 低流量：0.5-1.0 (50-100%)

3. **使用批量导出器 (Batch Exporters)** (默认)
    - Span 在导出前会进行批处理
    - 减少网络开销

4. **监控导出器健康状况**
    - 关注日志中的导出失败信息
    - 配置重试策略

## 故障排除

### 追踪未显示

1. **检查追踪是否已启用**：

```yaml
observability:
  tracing:
    enabled: true
```

1. **验证导出器端点**：

```bash
# 测试 OTLP 端点连接性
telnet jaeger 4317
```

1. **检查日志错误**：

```
Failed to export spans: connection refused
```

### 缺少 Span

1. **检查采样率**：

```yaml
sampling:
  type: "probabilistic"
  rate: 1.0  # 提高采样率以查看更多追踪
```

1. **验证代码中的 Span 创建**：

- Span 应在关键处理点创建
- 检查是否存在 nil context

### 内存占用过高

1. **降低采样率**：

```yaml
sampling:
  rate: 0.01  # 1% 采样
```

1. **验证批量导出器是否正常工作**：

- 检查导出间隔
- 监控队列长度

## 最佳实践

1. **开发时从 stdout 开始**
    - 易于验证追踪是否正常工作
    - 无外部依赖
2. **生产环境使用概率采样**
    - 平衡可见性和性能
    - 从 10% 开始并逐步调整
3. **设置有意义的服务名称**
    - 使用环境特定的名称
    - 包含版本信息
4. **为您自己的用例添加自定义属性**
    - 客户 ID
    - 部署区域
    - 特性标志 (Feature Flags)
5. **监控导出器健康状况**
    - 跟踪导出成功率
    - 对高失败率设置告警
6. **与指标关联**
    - 使用相同的服务名称
    - 在日志中交叉引用追踪 ID

## 与 vLLM 技术栈集成

### 未来增强

追踪实现旨在支持未来与 vLLM 后端的集成：

1. **追踪上下文传播** 到 vLLM
2. **跨路由和引擎的关联 Span**
3. **端到端延迟** 分析
4. **来自 vLLM 的 Token 级计时**

敬请关注 vLLM 集成的更新！

## 参考

- [OpenTelemetry Go SDK](https://github.com/open-telemetry/opentelemetry-go)
- [OpenInference 语义约定](https://github.com/Arize-ai/openinference)
- [Jaeger 文档](https://www.jaegertracing.io/docs/)
- [Grafana Tempo](https://grafana.com/oss/tempo/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
