---
title: 容器连通性故障排除
sidebar_label: 容器连通性
translation:
  source_commit: "2644304"
  source_file: "docs/troubleshooting/container-connectivity.md"
  outdated: false
---

本指南总结了我们在使用 Docker Compose 或 Kubernetes 运行 Router 时遇到的常见连通性问题及其解决方法。同时也涵盖了 Grafana 中的"无数据"问题以及如何验证完整的 metrics 链。

## 1. Backend Endpoint 使用 IPv4 地址

症状

- Router/Envoy 超时、5xx 错误，或 Prometheus 中出现"up/down"抖动。从容器/Pod 内部执行 curl 失败。

根本原因

- Backend 仅绑定到 127.0.0.1（从容器/Pod 无法访问）。
- 使用 IPv6 或解析为 IPv6 的主机名，而 IPv6 被禁用/阻止。
- 在 Router 配置中使用 localhost/127.0.0.1，这指向容器本身而非宿主机。

解决方法

- 确保 backend 绑定到所有接口：0.0.0.0。
- 在 Docker Compose 中，配置 Router 通过可达的 IPv4 地址调用宿主机。
  - 在 macOS 上，host.docker.internal 通常有效；如果无效，使用宿主机的局域网 IPv4 地址。
  - 在 Linux 或自定义网络上，使用您网络的 Docker 主机网关 IPv4。

示例：在宿主机上启动 vLLM

```bash
# 让 vLLM 监听所有接口
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 11434 \
  --served-model-name phi4
```

Router 配置示例（Docker Compose）

```yaml
# config/config.yaml（片段）
llm_backends:
  - name: phi4
    # 使用可达的 IPv4；替换为您宿主机的 IP
    address: http://172.28.0.1:11434
```

Kubernetes 推荐模式：使用 Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-vllm
spec:
  selector:
    app: my-vllm
  ports:
    - name: http
      port: 8000
      targetPort: 8000
```

然后 Router 配置使用：http://my-vllm.default.svc.cluster.local:8000

**提示**：从容器内部发现主机网关（主要适用于 Linux）

```bash
# 在容器/Pod 内部
ip route | awk '/default/ {print $3}'
```

## 2. 宿主机防火墙阻止容器/Pod 流量

症状

- 宿主机可以 curl backend，但容器/Pod 超时，直到防火墙开放。

解决方法

- macOS：系统设置 → 网络 → 防火墙。允许 backend 进程（如 Python/uvicorn）的传入连接，或临时禁用防火墙进行测试。
- Linux 示例：

```bash
# UFW（Ubuntu/Debian）
sudo ufw allow 11434/tcp
sudo ufw allow 11435/tcp

# firewalld（RHEL/CentOS/Fedora）
sudo firewall-cmd --add-port=11434/tcp --permanent
sudo firewall-cmd --add-port=11435/tcp --permanent
sudo firewall-cmd --reload
```

- 云主机：还需开放安全组/ACL 规则。

从容器/Pod 验证：

```bash
docker compose exec semantic-router curl -sS http://<IPv4>:11434/v1/models
```

## 3. Docker Compose：发布 Router 端口（不仅仅是 expose）

症状

- 无法从宿主机访问 /metrics 或 API。docker ps 显示没有发布的端口。

根本原因

- 仅使用 `expose` 只会将端口保持在 Compose 网络内部；不会发布到宿主机。

解决方法

- 使用 `ports:` 映射所需端口。

docker-compose.yml 片段示例（来自 `deploy/docker-compose/docker-compose.yml`）

```yaml
services:
  semantic-router:
    # ...
    ports:
      - "9190:9190" # Prometheus /metrics
      - "50051:50051" # gRPC/HTTP API（使用您的实际服务端口）
```

从宿主机验证：

```bash
curl -sS http://localhost:9190/metrics | head -n 5
```

## 4. Grafana 仪表板显示"无数据"

常见原因及解决方法

- 指标尚未产生
  - 某些面板在代码路径被触发之前为空。示例：
    - 成本：`llm_model_cost_total{currency="USD"}` 仅在记录成本时增长。
    - 拒绝：`llm_request_errors_total{reason="pii_policy_denied"|"jailbreak_block"}` 仅在 policy 阻止请求时增长。
  - 生成相关流量或启用过滤器/策略以查看数据。

- 面板查询细节
  - 分类条形图通常需要即时查询。
  - 分位数需要直方图桶。

有用的 PromQL 示例（用于 Explore）

```promql
# 类别分类（即时）
sum by (category) (llm_category_classifications_count)

# 成本速率（USD/秒）
sum by (model) (rate(llm_model_cost_total{currency="USD"}[5m]))

# 每个模型的拒绝数
sum by (model) (rate(llm_request_errors_total{reason=~"pii_policy_denied|jailbreak_block"}[5m]))

# 拒绝率百分比
100 * sum by (model) (rate(llm_request_errors_total{reason=~"pii_policy_denied|jailbreak_block"}[5m]))
  / sum by (model) (rate(llm_model_requests_total[5m]))

# 延迟 p95
histogram_quantile(0.95, sum by (le) (rate(llm_model_completion_latency_seconds_bucket[5m])))
```

Prometheus 抓取配置（验证目标为 UP）

```yaml
scrape_configs:
  - job_name: semantic-router
    static_configs:
      - targets: ["semantic-router:9190"]

  - job_name: envoy
    metrics_path: /stats/prometheus
    static_configs:
      - targets: ["envoy-proxy:19000"]
```

时间范围和刷新

- 选择包含您最近流量的时间窗口（最近 5-15 分钟），并在发送测试请求后刷新仪表板。

## 快速检查清单

- Backend 监听 0.0.0.0；Router 使用可达的 IPv4 地址（或解析为 IPv4 的 k8s Service DNS）。
- 宿主机防火墙允许 backend 端口；如适用，云安全组/ACL 已开放。
- 在 Docker Compose 中，Router 端口已发布（如 9190 用于 /metrics，服务端口用于 API）。
- `semantic-router:9190` 和 `envoy-proxy:19000` 的 Prometheus target 为 UP。
- 发送触发您期望的 metrics（成本/拒绝）的流量，并根据需要调整面板查询模式（instant vs. range）。
