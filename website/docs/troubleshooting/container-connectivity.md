---
title: Container Connectivity Troubleshooting
sidebar_label: Container Connectivity
---

This guide summarizes common connectivity issues we hit when running the router with Docker Compose or Kubernetes and how we fixed them. It also covers the “No data” problem in Grafana and how to validate the full metrics chain.

## 1. Use IPv4 addresses for backend endpoints

Symptoms

- Router/Envoy timeouts, 5xx, or “up/down” flapping in Prometheus. Curl from inside containers/pods fails.

Root causes

- Backend bound only to 127.0.0.1 (not reachable from containers/pods).
- Using IPv6 or hostnames that resolve to IPv6 where IPv6 is disabled/blocked.
- Using localhost/127.0.0.1 in the router config, which refers to the container itself, not the host.

Fixes

- Ensure backends bind to all interfaces: 0.0.0.0.
- In Docker Compose, configure the router to call the host via a reachable IPv4 address.
  - On macOS, host.docker.internal usually works; if not, use the host’s LAN IPv4 address.
  - On Linux or custom networks, use the Docker host gateway IPv4 for your network.

Example: start vLLM on the host

```bash
# Make vLLM listen on all interfaces
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 11434 \
  --served-model-name phi4
```

Router config example (Docker Compose)

```yaml
# config/config.yaml (snippet)
llm_backends:
  - name: phi4
    # Use a reachable IPv4; replace with your host’s IP
    address: http://172.28.0.1:11434
```

Kubernetes recommended pattern: use a Service

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

Router config then uses: http://my-vllm.default.svc.cluster.local:8000

**Tip**: discover the host gateway from inside a container (mostly Linux)

```bash
# Inside the container/pod
ip route | awk '/default/ {print $3}'
```

## 2. Host firewall blocking container/pod traffic

Symptoms

- Host can curl the backend, but containers/pods time out until the firewall is opened.

Fixes

- macOS: System Settings → Network → Firewall. Allow incoming connections for the backend process (e.g., Python/uvicorn) or temporarily disable the firewall to test.
- Linux examples:

```bash
# UFW (Ubuntu/Debian)
sudo ufw allow 11434/tcp
sudo ufw allow 11435/tcp

# firewalld (RHEL/CentOS/Fedora)
sudo firewall-cmd --add-port=11434/tcp --permanent
sudo firewall-cmd --add-port=11435/tcp --permanent
sudo firewall-cmd --reload
```

- Cloud hosts: also open security group/ACL rules.

Validate from the container/pod:

```bash
docker compose exec semantic-router curl -sS http://<IPv4>:11434/v1/models
```

## 3. Docker Compose: publish the router’s ports (not just expose)

Symptoms

- Can’t access /metrics or API from the host. docker ps shows no published ports.

Root cause

- Using `expose` only keeps ports internal to the Compose network; it doesn’t publish to the host.

Fix

- Map the needed ports with `ports:`.

Example docker-compose.yml snippet (from `deploy/docker-compose/docker-compose.yml` after relocation)

```yaml
services:
  semantic-router:
    # ...
    ports:
      - "9190:9190" # Prometheus /metrics
      - "50051:50051" # gRPC/HTTP API (use your actual service port)
```

Validate from the host:

```bash
curl -sS http://localhost:9190/metrics | head -n 5
```

## 4. Grafana dashboard shows “No data”

Common causes and fixes

- Metrics not emitted yet
  - Some panels are empty until code paths are hit. Examples:
    - Cost: `llm_model_cost_total{currency="USD"}` grows only when cost is recorded.
    - Refusals: `llm_request_errors_total{reason="pii_policy_denied"|"jailbreak_block"}` grows only when policies block requests.
  - Generate relevant traffic or enable filters/policies to see data.

- Panel query nuances
  - Classification bar gauge often needs instant query.
  - Quantiles require histogram buckets.

Useful PromQL examples (for Explore)

```promql
# Category classification (instant)
sum by (category) (llm_category_classifications_count)

# Cost rate (USD/sec)
sum by (model) (rate(llm_model_cost_total{currency="USD"}[5m]))

# Refusals per model
sum by (model) (rate(llm_request_errors_total{reason=~"pii_policy_denied|jailbreak_block"}[5m]))

# Refusal rate percentage
100 * sum by (model) (rate(llm_request_errors_total{reason=~"pii_policy_denied|jailbreak_block"}[5m]))
  / sum by (model) (rate(llm_model_requests_total[5m]))

# Latency p95
histogram_quantile(0.95, sum by (le) (rate(llm_model_completion_latency_seconds_bucket[5m])))
```

Prometheus scrape config (verify targets are UP)

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

Time range & refresh

- Select a window that includes your recent traffic (Last 5–15 minutes) and refresh the dashboard after sending test requests.

## Quick checklist

- Backends listen on 0.0.0.0; router uses a reachable IPv4 address (or k8s Service DNS that resolves to IPv4).
- Host firewall allows the backend ports; cloud SG/ACL opened if applicable.
- In Docker Compose, router ports are published (e.g., 9190 for /metrics, service port for API).
- Prometheus targets for `semantic-router:9190` and `envoy-proxy:19000` are UP.
- Send traffic that triggers the metrics you expect (cost/refusals) and adjust panel query mode (instant vs. range) where needed.
