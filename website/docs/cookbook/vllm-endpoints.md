---
title: vLLM Endpoints Configuration
sidebar_label: vLLM Endpoints
---

# vLLM Endpoints Configuration

This guide provides quick configuration recipes for vLLM backend endpoints and load balancing. Use these patterns to set up single or multi-endpoint deployments with weighted traffic distribution.

## Basic Endpoint Definition

Define a single vLLM endpoint:

```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: "172.28.0.20" # IPv4 address
    port: 8002
    weight: 1
```

> See: [config.yaml#vllm_endpoints](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L43-L51).

:::caution
The `address` field must be a valid IP address (IPv4 or IPv6).

- ✅ Supported: `127.0.0.1`, `192.168.1.1`, `::1`, `2001:db8::1`
- ❌ Not supported: domain names, protocol prefixes (`http://`), paths, or ports in the address field

:::

## Multiple Endpoints with Load Balancing

Configure multiple endpoints with weighted distribution:

```yaml
vllm_endpoints:
  - name: "primary"
    address: "10.0.0.10"
    port: 8000
    weight: 3 # Receives 3x traffic

  - name: "secondary"
    address: "10.0.0.11"
    port: 8000
    weight: 1 # Receives 1x traffic
```

## Map Models to Specific Endpoints

Route specific models to preferred endpoints:

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

> See: [config.yaml#preferred_endpoints](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L53-L56) AND [config.go endpoints](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/config.go).

## IPv6 Endpoint Configuration

Use IPv6 addresses for endpoints:

```yaml
vllm_endpoints:
  - name: "ipv6_endpoint"
    address: "2001:db8::1"
    port: 8000
    weight: 1
```

## Docker Compose Network Endpoints

When using Docker Compose, use container IP or service name resolution:

```yaml
# In config.yaml
vllm_endpoints:
  - name: "llm-katan"
    address: "172.28.0.20" # Static IP assigned in docker-compose.yml
    port: 8002
    weight: 1
```

```yaml
# In docker-compose.yml
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

> See: [config.yaml#vllm_endpoints](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L43-L51) AND [docker-compose.yml](https://github.com/vllm-project/semantic-router/blob/main/deploy/docker-compose/docker-compose.yml).

## Kubernetes Endpoints

For Kubernetes deployments, use Service ClusterIP or Pod IP:

```yaml
vllm_endpoints:
  - name: "vllm-svc"
    address: "10.96.100.50" # ClusterIP of vLLM Service
    port: 8000
    weight: 1
```

## High Availability Setup

Configure multiple endpoints for failover:

```yaml
vllm_endpoints:
  # Primary datacenter
  - name: "dc1-primary"
    address: "10.1.0.10"
    port: 8000
    weight: 2

  - name: "dc1-secondary"
    address: "10.1.0.11"
    port: 8000
    weight: 1

  # Secondary datacenter (lower weight for DR)
  - name: "dc2-primary"
    address: "10.2.0.10"
    port: 8000
    weight: 1
```

## Endpoint Validation Checklist

Before deploying, verify:

| Check              | Command                                  |
| ------------------ | ---------------------------------------- |
| IP is reachable    | `ping <address>`                         |
| Port is open       | `nc -zv <address> <port>`                |
| vLLM is responding | `curl http://<address>:<port>/health`    |
| Model is loaded    | `curl http://<address>:<port>/v1/models` |

## Common Mistakes

### ❌ Using Domain Names

```yaml
# WRONG - domain names not supported
vllm_endpoints:
  - name: "endpoint1"
    address: "vllm.example.com" # ❌ Won't work
```

### ❌ Including Protocol or Port in Address

```yaml
# WRONG - no protocol prefix or port in address
vllm_endpoints:
  - name: "endpoint1"
    address: "http://10.0.0.10:8000" # ❌ Wrong format
```

### ✅ Correct Format

```yaml
# CORRECT
vllm_endpoints:
  - name: "endpoint1"
    address: "10.0.0.10" # ✅ IP only
    port: 8000 # ✅ Port separate
```
