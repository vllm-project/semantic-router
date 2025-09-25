---
sidebar_position: 3
---

# Deployment Quickstart

This unified guide helps you quickly run Semantic Router locally (Docker Compose) or in a cluster (Kubernetes) and explains when to choose each path. Both share the same configuration concepts: Docker is ideal for rapid iteration and demos, while Kubernetes is suited for long‑running workloads, elasticity, and upcoming Operator / CRD scenarios.

---

## 1. Choosing a Path (When to Use Which)

| Scenario / Goal                             | Recommended Path                 | Why                                                                              |
| ------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------- |
| Local dev, quickest iteration, hacking code | Docker Compose                   | One command starts router + Envoy + (optionally) mock vLLM + observability stack |
| Demo with dashboard quickly                 | Docker Compose (testing profile) | Bundled Prometheus + Grafana + mock responses                                    |
| Team shared staging / pre‑prod              | Kubernetes                       | Declarative config, rolling upgrades, persistent model volume                    |
| Performance, scalability, autoscaling       | Kubernetes                       | HPA, scheduling, resource isolation                                              |
| Future Operator / CRD driven config         | Kubernetes                       | Native controller pattern                                                        |

You can seamlessly reuse the same configuration concepts in both paths.

---

## 2. Common Prerequisites (Both Paths)

Clone repo：

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
```

Download classification models (≈1.5GB, first run only):

```bash
make download-models
```

Includes: Category / PII / Jailbreak (ModernBERT). The similarity BERT model is pulled remotely by default; you can switch to a local path in `config/config.yaml`.

---

## 3. Path A: Docker Compose Quick Start

### Requirements

- Docker Engine
- Docker Compose v2 (`docker compose version` should work)
- Ensure ports 8801, 50051, 19000 (and 3000/9090 if using full observability) are free

Install Compose plugin (if missing):

```bash
# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y docker-compose-plugin

# RHEL / CentOS / Fedora
sudo yum update -y && sudo yum install -y docker-compose-plugin

# Verify
docker compose version
```

### Start Services

```bash
# Core (router + envoy)
docker compose up --build

# Detached (recommended once OK)
docker compose up -d --build

# Include mock vLLM + testing profile (points router to mock endpoint)
CONFIG_FILE=/app/config/config.testing.yaml \
  docker compose --profile testing up --build
```

### Verify

- gRPC: `localhost:50051`
- Envoy HTTP: `http://localhost:8801`
- Envoy Admin: `http://localhost:19000`

Optional (if observability stack enabled):

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin / admin)

### Common Operations

```bash
# Status
docker compose ps

# Logs
docker compose logs -f semantic-router

# Exec
docker compose exec semantic-router bash

# Recreate after config change
docker compose up -d --build

# Stop / clean
docker compose down
```

### Config Overrides

Switch to the testing config:

```bash
CONFIG_FILE=/app/config/config.testing.yaml docker compose up --build
```

If Hugging Face access fails, add the following to the `semantic-router` service in `docker-compose.yml`:

```yaml
services:
  semantic-router:
    dns: ["1.1.1.1", "8.8.8.8"]
    environment:
      http_proxy: "http://YOUR_PROXY"
      https_proxy: "http://YOUR_PROXY"
      no_proxy: "localhost,127.0.0.1"
```

---

## 4. Path B: Kubernetes Quick Start

### Requirements

- Kubernetes cluster (kind / k3d / minikube / real)
- `kubectl` access
- Optional: Prometheus (Operator) for metrics scraping

### Deploy (Kustomize)

```bash
kubectl apply -k deploy/kubernetes/

# Wait for pod
kubectl -n semantic-router get pods
```

Manifests create:

- Deployment (main container + init model downloader)
- Service `semantic-router` (gRPC 50051)
- Service `semantic-router-metrics` (metrics 9190)
- ConfigMap (base config)
- PVC (model cache)

### Port Forward (Ad-hoc)

```bash
kubectl -n semantic-router port-forward svc/semantic-router 50051:50051 &
kubectl -n semantic-router port-forward svc/semantic-router-metrics 9190:9190 &
```

### Observability (Summary)

- Add a `ServiceMonitor` or a static scrape rule
- Import `deploy/llm-router-dashboard.json` (see `observability.md`)

### Updating Config

`deploy/kubernetes/config.yaml` updated：

```bash
kubectl apply -k deploy/kubernetes/
kubectl -n semantic-router rollout restart deploy/semantic-router
```

### Typical Customizations

| Goal               | Change                                              |
| ------------------ | --------------------------------------------------- |
| Scale horizontally | `kubectl scale deploy/semantic-router --replicas=N` |
| Resource tuning    | Edit `resources:` in `deployment.yaml`              |
| Add HTTP readiness | Switch TCP probe -> HTTP `/health` (port 8080)      |
| PVC size           | Adjust storage request in PVC manifest              |
| Metrics scraping   | Add ServiceMonitor / scrape rule                    |

---

## 5. Feature Comparison

| Capability               | Docker Compose      | Kubernetes                                     |
| ------------------------ | ------------------- | ---------------------------------------------- |
| Startup speed            | Fast (seconds)      | Depends on cluster/image pull                  |
| Config reload            | Manual recreate     | Rolling restart / future Operator / hot reload |
| Model caching            | Host volume/bind    | PVC persistent across pods                     |
| Observability            | Bundled stack       | Integrate existing stack                       |
| Autoscaling              | Manual              | HPA / custom metrics                           |
| Isolation / multi-tenant | Single host network | Namespaces / RBAC                              |
| Rapid hacking            | Minimal friction    | YAML overhead                                  |
| Production lifecycle     | Basic               | Full (probes, rollout, scaling)                |

---

## 6. Troubleshooting (Unified)

### HF model download failure / DNS errors

Log example: `Dns Failed: resolve huggingface.co`.

Fix:

```yaml
services:
  semantic-router:
    dns: ["1.1.1.1", "8.8.8.8"]
    environment:
      http_proxy: http://YOUR_PROXY
      https_proxy: http://YOUR_PROXY
      no_proxy: localhost,127.0.0.1
```

Or use a local model:

```yaml
bert_model:
  model_id: "models/sentence-transformers/all-MiniLM-L12-v2"
  threshold: 0.6
  use_cpu: true
```

Then `docker compose up -d --build`。

### Envoy / Router request failures

- Ensure the correct config is used: the testing profile requires passing `CONFIG_FILE=...config.testing.yaml`
- Check if mock vLLM is healthy: `docker compose ps` / view logs
- Send a smoke test:

```bash
curl -X POST http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
```

### (Docker) No metrics data

- Ensure service names match those in `config/prometheus.yaml`
- Generate traffic to trigger classifications / inference

### (K8s) Init container repeatedly failing

```bash
kubectl -n semantic-router logs pod/<pod> -c model-downloader
```

Check network / DNS / proxy environment variables.

### (K8s) Metrics not scraped

- `kubectl get ep semantic-router-metrics -n semantic-router`
- ServiceMonitor 选择器是否匹配 `service: metrics` 标签

### Port conflicts

Adjust external port mappings in `docker-compose.yml`, or free local ports 8801 / 50051 / 19000.

Extra tip: If you use the testing profile, also pass the testing config so the router targets the mock service:

```bash
CONFIG_FILE=/app/config/config.testing.yaml docker compose --profile testing up --build
```

**2. Envoy/Router up but requests fail**

- Ensure `mock-vllm` is healthy (testing profile only):
  - `docker compose ps` should show mock-vllm healthy; logs show 200 on `/health`.
- Verify the router config in use:
  - Router logs print `Starting vLLM Semantic Router ExtProc with config: ...`. If it shows `/app/config/config.yaml` while testing, you forgot `CONFIG_FILE`.
- Basic smoke test via Envoy (OpenAI-compatible):
  - Send a POST to `http://localhost:8801/v1/chat/completions` with `{"model":"auto", "messages":[{"role":"user","content":"hi"}]}` and check that the mock responds with `[mock-openai/gpt-oss-20b]` content when testing profile is active.

**3. DNS problems inside containers**

If DNS is flaky in your Docker environment, add DNS servers to the `semantic-router` service in `docker-compose.yml`:

```yaml
services:
  semantic-router:
    # ...
    dns:
      - 1.1.1.1
      - 8.8.8.8
```

For corporate proxies, set `http_proxy`, `https_proxy`, and `no_proxy` in the service `environment`.

Make sure 8801, 50051, 19000 are not bound by other processes. Adjust ports in `docker-compose.yml` if needed.

**4. (K8s) Init container repeatedly failing**

Check logs:

```bash
kubectl -n semantic-router logs pod/<pod-name> -c model-downloader
```

Add network/DNS env vars or mirror settings; ensure Hugging Face reachable.

**5. (K8s) No metrics scraped**

- Confirm `Service/semantic-router-metrics` has endpoints: `kubectl -n semantic-router get ep semantic-router-metrics`
- Verify ServiceMonitor label selector or static scrape rule regex.

---

## 7. Next Steps

- Configure endpoints: see `installation.md` & `configuration.md`.
- Add observability details: see `observability.md`.
- Track roadmap & Operator work: project README / issues.

Happy routing! Choose Compose for speed or Kubernetes for scale — both share the same core binary & config schema.
