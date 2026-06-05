---
title: 指标与监控
sidebar_label: 指标
translation:
  source_commit: "a6e1910"
  source_file: "docs/tutorials/observability/metrics.md"
  outdated: false
---

# 指标与监控

使用 Prometheus 和 Grafana 进行Semantic Router 的指标收集和可视化。

---

## 1. 指标与端点

| 组件                | 端点                  | 说明                                      |
| ------------------------ | ------------------------- | ------------------------------------------ |
| 路由指标           | `:9190/metrics`           | Prometheus 格式（标志：`--metrics-port`） |
| 路由健康检查            | `:8080/health`            | HTTP 就绪/存活探测                    |
| Envoy 指标（可选） | `:19000/stats/prometheus` | 如果启用了 Envoy                        |

**配置位置**：`tools/observability/`  
**仪表板**：`tools/observability/llm-router-dashboard.json`

---

## 2. 本地模式（路由在宿主机上运行）

路由在宿主机上原生运行，可观测性组件在 Docker 中。

### 快速开始

```bash
# 启动路由
make run-router

# 启动可观测性
make o11y-local
```

**访问：**

- Prometheus：http://localhost:9090
- Grafana：http://localhost:3000 (admin/admin)

**验证目标：**

```bash
# 检查 Prometheus 抓取 localhost:9190
open http://localhost:9090/targets
```

**停止：**

```bash
make stop-observability
```

### 配置

所有配置在 `tools/observability/` 中：

- `prometheus.yaml` - 从 `ROUTER_TARGET` 环境变量抓取目标（默认：`localhost:9190`）
- `grafana-datasource.yaml` - 指向 `localhost:9090`
- `grafana-dashboard.yaml` - 仪表板配置
- `llm-router-dashboard.json` - 仪表板定义

### 故障排除

| 问题         | 修复                                     |
| ------------- | --------------------------------------- |
| 目标 DOWN   | 启动路由：`make run-router`         |
| 无指标    | 生成流量，检查 `:9190/metrics` |
| 端口冲突 | 更改端口或停止冲突服务 |

---

## 3. Docker Compose 模式

所有服务在 Docker 容器中。

### 快速开始

```bash
# 启动完整堆栈（包括可观测性）
docker compose -f deploy/docker-compose/docker-compose.yml up --build

# 或使用测试配置文件
docker compose -f deploy/docker-compose/docker-compose.yml --profile testing up --build
```

**访问：**

- Prometheus：http://localhost:9090
- Grafana：http://localhost:3000 (admin/admin)

**预期目标：**

- `semantic-router:9190`
- `envoy-proxy:19000`（可选）

### 配置

与本地模式相同的配置（`tools/observability/`），但：

- `ROUTER_TARGET=semantic-router:9190`
- `PROMETHEUS_URL=prometheus:9090`
- 使用 `semantic-network` 桥接网络

---

## 4. Kubernetes 模式

用于 K8s 集群的生产就绪 Prometheus + Grafana。

> **命名空间**：`vllm-semantic-router-system`

### 组件

| 组件  | 用途                               | 位置                                       |
| ---------- | ------------------------------------- | ---------------------------------------------- |
| Prometheus | 抓取路由指标，15 天保留 | `deploy/kubernetes/observability/prometheus/`  |
| Grafana    | 仪表板可视化               | `deploy/kubernetes/observability/grafana/`     |
| Ingress    | 可选的外部访问              | `deploy/kubernetes/observability/ingress.yaml` |

### 部署

```bash
# 应用清单
kubectl apply -k deploy/kubernetes/observability/

# 验证
kubectl get pods -n vllm-semantic-router-system
```

### 访问

**端口转发：**

```bash
kubectl port-forward svc/prometheus 9090:9090 -n vllm-semantic-router-system
kubectl port-forward svc/grafana 3000:3000 -n vllm-semantic-router-system
```

**Ingress：** 使用您的域名和 TLS 自定义 `ingress.yaml`

### 关键配置

**Prometheus** 使用 Kubernetes 服务发现：

```yaml
scrape_configs:
  - job_name: semantic-router
    kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names: [vllm-semantic-router-system]
```

**Grafana** 凭据（生产环境中请更改）：

```bash
kubectl create secret generic grafana-admin \
  --namespace vllm-semantic-router-system \
  --from-literal=admin-user=admin \
  --from-literal=admin-password='your-password'
```

---

## 5. 关键指标

| 指标                                  | 类型      | 描述              |
| --------------------------------------- | --------- | ------------------------ |
| `llm_category_classifications_count`    | counter   | 类别分类次数 |
| `llm_model_completion_tokens_total`     | counter   | 每个模型的 Token 数         |
| `llm_model_routing_modifications_total` | counter   | 模型路由更改次数    |
| `llm_model_completion_latency_seconds`  | histogram | 完成延迟       |

**示例查询：**

```promql
rate(llm_model_completion_tokens_total[5m])
histogram_quantile(0.95, rate(llm_model_completion_latency_seconds_bucket[5m]))
```

---

## 6. 窗口化模型指标（负载均衡）

增强的时间窗口指标用于模型性能跟踪，适用于 Kubernetes 环境中模型选择比端点选择更重要的负载均衡决策。

### 配置

在 `config.yaml` 中启用窗口化指标：

```yaml
observability:
  metrics:
    windowed_metrics:
      enabled: true
      time_windows: ["1m", "5m", "15m", "1h", "24h"]
      update_interval: "10s"
      model_metrics: true
      queue_depth_estimation: true
      max_models: 100
```

### 模型级别指标

| 指标                                      | 类型  | 标签                       | 描述                           |
| ------------------------------------------- | ----- | ---------------------------- | ------------------------------------- |
| `llm_model_latency_windowed_seconds`        | gauge | model, time_window           | 每个时间窗口的平均延迟       |
| `llm_model_requests_windowed_total`         | gauge | model, time_window           | 每个时间窗口的请求数         |
| `llm_model_tokens_windowed_total`           | gauge | model, token_type, time_window | 每个窗口的 Token 吞吐量         |
| `llm_model_utilization_percentage`          | gauge | model, time_window           | 估计利用率百分比      |
| `llm_model_queue_depth_estimated`           | gauge | model                        | 当前估计队列深度         |
| `llm_model_error_rate_windowed`             | gauge | model, time_window           | 每个时间窗口的错误率            |
| `llm_model_latency_p50_windowed_seconds`    | gauge | model, time_window           | 每个时间窗口的 P50 延迟           |
| `llm_model_latency_p95_windowed_seconds`    | gauge | model, time_window           | 每个时间窗口的 P95 延迟           |
| `llm_model_latency_p99_windowed_seconds`    | gauge | model, time_window           | 每个时间窗口的 P99 延迟           |

### 示例查询

```promql
# 模型过去 5 分钟的平均延迟
llm_model_latency_windowed_seconds{model="gpt-4", time_window="5m"}

# 跨模型的 P95 延迟比较
llm_model_latency_p95_windowed_seconds{time_window="15m"}

# 每个模型的 Token 吞吐量
llm_model_tokens_windowed_total{token_type="completion", time_window="1h"}

# 用于负载均衡决策的当前队列深度
llm_model_queue_depth_estimated{model="gpt-4"}

# 错误率监控
llm_model_error_rate_windowed{time_window="5m"} > 0.05
```

### 用例

1. **负载均衡**：使用队列深度和延迟指标将请求路由到负载较低的模型
2. **性能监控**：跟踪跨时间窗口的 P95/P99 延迟趋势
3. **容量规划**：监控利用率百分比以确定何时扩展模型
4. **告警**：对特定时间窗口内的错误率或延迟峰值设置告警

---

## 7. 故障排除

| 问题           | 检查               | 修复                                                   |
| --------------- | ------------------- | ----------------------------------------------------- |
| 目标 DOWN     | Prometheus /targets | 验证路由正在运行并暴露 `:9190/metrics` |
| 无指标      | 生成流量    | 通过路由发送请求                          |
| 仪表板为空 | Grafana 数据源  | 检查 Prometheus URL 配置                    |

---
