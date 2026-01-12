---
title: vLLM 端点配置
sidebar_label: vLLM 端点
translation:
  source_commit: "1d1439a"
  source_file: "docs/cookbook/vllm-endpoints.md"
  outdated: false
---

# vLLM 端点配置

本指南提供了 vLLM backend endpoint 和 load balancing 的快速配置方案。使用这些模式设置单 endpoint 或多 endpoint 部署，并进行加权流量分配。

## 基本 Endpoint 定义

定义单个 vLLM endpoint：

```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: "172.28.0.20" # IPv4 地址
    port: 8002
    weight: 1
```

> 参见：[config.yaml#vllm_endpoints](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L43-L51)。

:::caution
`address` 字段必须是有效的 IP 地址（IPv4 或 IPv6）。

- ✅ 支持：`127.0.0.1`、`192.168.1.1`、`::1`、`2001:db8::1`
- ❌ 不支持：域名、协议前缀（`http://`）、路径或地址字段中的端口

:::

## 带 Load Balancing 的多 Endpoint

配置带加权分配的多个 endpoint：

```yaml
vllm_endpoints:
  - name: "primary"
    address: "10.0.0.10"
    port: 8000
    weight: 3 # 接收 3 倍流量

  - name: "secondary"
    address: "10.0.0.11"
    port: 8000
    weight: 1 # 接收 1 倍流量
```

## 将 Model 映射到特定 Endpoint

将特定 model 路由到首选 endpoint：

```yaml
vllm_endpoints:
  - name: "gpu_cluster_a"
    address: "10.0.1.10"
    port: 8000
    weight: 1

  - name: "gpu_cluster_b"
    address: "10.0.2.10"
    port: 8000
    weight: 1

model_config:
  "qwen3":
    reasoning_family: "qwen3"
    preferred_endpoints: ["gpu_cluster_a"]

  "llama":
    reasoning_family: "llama"
    preferred_endpoints: ["gpu_cluster_b"]
```

> 参见：[config.yaml#preferred_endpoints](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L53-L56) 和 [config.go endpoints](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/config.go)。

## IPv6 Endpoint 配置

使用 IPv6 地址作为 endpoint：

```yaml
vllm_endpoints:
  - name: "ipv6_endpoint"
    address: "2001:db8::1"
    port: 8000
    weight: 1
```

## Docker Compose 网络 Endpoint

使用 Docker Compose 时，使用容器 IP 或 service name 解析：

```yaml
# 在 config.yaml 中
vllm_endpoints:
  - name: "llm-katan"
    address: "172.28.0.20" # docker-compose.yml 中分配的静态 IP
    port: 8002
    weight: 1
```

```yaml
# 在 docker-compose.yml 中
services:
  llm-service:
    networks:
      app-network:
        ipv4_address: 172.28.0.20

networks:
  app-network:
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

> 参见：[config.yaml#vllm_endpoints](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L43-L51) 和 [docker-compose.yml](https://github.com/vllm-project/semantic-router/blob/main/deploy/docker-compose/docker-compose.yml)。

## Kubernetes Endpoint

对于 Kubernetes 部署，使用 Service ClusterIP 或 Pod IP：

```yaml
vllm_endpoints:
  - name: "vllm-svc"
    address: "10.96.100.50" # vLLM Service 的 ClusterIP
    port: 8000
    weight: 1
```

## 高可用设置

配置多个 endpoint 以实现 failover：

```yaml
vllm_endpoints:
  # 主数据中心
  - name: "dc1-primary"
    address: "10.1.0.10"
    port: 8000
    weight: 2

  - name: "dc1-secondary"
    address: "10.1.0.11"
    port: 8000
    weight: 1

  # 备用数据中心（较低权重用于灾备）
  - name: "dc2-primary"
    address: "10.2.0.10"
    port: 8000
    weight: 1
```

## Endpoint 验证 Checklist

部署前验证：

| 检查项              | 命令                                  |
| ------------------ | ---------------------------------------- |
| IP 可达    | `ping <address>`                         |
| 端口开放       | `nc -zv <address> <port>`                |
| vLLM 响应正常 | `curl http://<address>:<port>/health`    |
| 模型已加载    | `curl http://<address>:<port>/v1/models` |

## 常见错误

### ❌ 使用域名

```yaml
# 错误 - 不支持域名
vllm_endpoints:
  - name: "endpoint1"
    address: "vllm.example.com" # ❌ 不会生效
```

### ❌ 在地址中包含协议或端口

```yaml
# 错误 - 地址中不能有协议前缀或端口
vllm_endpoints:
  - name: "endpoint1"
    address: "http://10.0.0.10:8000" # ❌ 格式错误
```

### ✅ 正确格式

```yaml
# 正确
vllm_endpoints:
  - name: "endpoint1"
    address: "10.0.0.10" # ✅ 仅 IP
    port: 8000 # ✅ 端口单独设置
```
