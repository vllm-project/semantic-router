---
sidebar_position: 3
---

# Containerized Deployment

This unified guide helps you quickly run Semantic Router locally (Docker Compose) or in a cluster (Kubernetes) and explains when to choose each path.Both share the same configuration concepts: **Docker Compose** is ideal for rapid iteration and demos, while **Kubernetes** is suited for long‑running workloads, elasticity, and upcoming Operator / CRD scenarios.

## Choosing a Path

**Docker Compose path** = semantic-router + Envoy proxy + optional mock vLLM (testing profile) + Prometheus + Grafana. It gives you an end-to-end local playground with minimal friction.

**Kubernetes path** (current manifests) = ONLY the semantic-router Deployment (gRPC + metrics), a PVC for model cache, its ConfigMap, and two Services (gRPC + metrics). It does NOT yet bundle Envoy, a real LLM inference backend, Istio, or any CRDs/Operator.

| Scenario / Goal                             | Recommended Path                 | Why                                                                              |
| ------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------- |
| Local dev, quickest iteration, hacking code | Docker Compose                   | One command starts router + Envoy + (optionally) mock vLLM + observability stack |
| Demo with dashboard quickly                 | Docker Compose (testing profile) | Bundled Prometheus + Grafana + mock responses                                    |
| Team shared staging / pre‑prod              | Kubernetes                       | Declarative config, rolling upgrades, persistent model volume                    |
| Performance, scalability, autoscaling       | Kubernetes                       | HPA, scheduling, resource isolation                                              |
| Future Operator / CRD driven config         | Kubernetes                       | Native controller pattern                                                        |

You can seamlessly reuse the same configuration concepts in both paths.

---

## Common Prerequisites

- **Docker Engine:** see more in [Docker Engine Installation](https://docs.docker.com/engine/install/)

- **Clone repo：**

  ```bash
  git clone https://github.com/vllm-project/semantic-router.git
  cd semantic-router
  ```

- **Download classification models (≈1.5GB, first run only):**

  ```bash
  make download-models
  ```

  This downloads the classification models used by the router:

  - Category classifier (ModernBERT-base)
  - PII classifier (ModernBERT-base)
  - Jailbreak classifier (ModernBERT-base)

---

## Path A: Docker Compose Quick Start

### Requirements

- Docker Compose v2 (`docker compose` command, not the legacy `docker-compose`)

  Install Docker Compose Plugin (if missing), see more in [Docker Compose Plugin Installation](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

  ```bash
  # For Debian / Ubuntu
  sudo apt-get update 
  sudo apt-get install -y docker-compose-plugin

  # For RHEL / CentOS / Fedora
  sudo yum update -y 
  sudo yum install -y docker-compose-plugin
  
  # Verify
  docker compose version
  ```

- Ensure ports 8801, 50051, 19000, 3000 and 9090 are free

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
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin` for first login)

### Common Operations

```bash
# View service status
docker compose ps

# Follow logs for the router service
docker compose logs -f semantic-router

# Exec into the router container
docker compose exec semantic-router bash

# Recreate after config change
docker compose up -d --build

# Stop and clean up containers
docker compose down
```

---

## Path B: Kubernetes Quick Start

### Requirements

- Kubernetes cluster
  - [Kubernetes Official docs](https://kubernetes.io/docs/home/)
  - [kind (local clusters)](https://kind.sigs.k8s.io/)
  - [k3d (k3s in Docker)](https://k3d.io/)
  - [minikube](https://minikube.sigs.k8s.io/docs/)
- [`kubectl`](https://kubernetes.io/docs/tasks/tools/)access (CLI)
- *Optional: Prometheus metrics stack (e.g. [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator))*
- *(Planned / not yet merged) Service Mesh or advanced gateway:*
  - *[Istio](https://istio.io/latest/docs/setup/getting-started/) / [Kubernetes Gateway API](https://gateway-api.sigs.k8s.io/)*
- Separate deployment of **Envoy** (or another gateway) + real **LLM endpoints** (follow [Installation guide](https://vllm-semantic-router.com/docs/getting-started/installation)).
  - Replace placeholder IPs in `deploy/kubernetes/config.yaml` once services exist.

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

## Feature Comparison

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

## Troubleshooting (Unified)

### HF model download failure / DNS errors
Log example: `Dns Failed: resolve huggingface.co`. See solutions in [Network Tips](https://vllm-semantic-router.com/docs/troubleshooting/network-tips/)

### Port conflicts

Adjust external port mappings in `docker-compose.yml`, or free local ports 8801 / 50051 / 19000.

Extra tip: If you use the testing profile, also pass the testing config so the router targets the mock service:

```bash
CONFIG_FILE=/app/config/config.testing.yaml docker compose --profile testing up --build
```

### Envoy/Router up but requests fail

- Ensure `mock-vllm` is healthy (testing profile only):
  - `docker compose ps` should show mock-vllm healthy; logs show 200 on `/health`.
- Verify the router config in use:
  - Router logs print `Starting vLLM Semantic Router ExtProc with config: ...`. If it shows `/app/config/config.yaml` while testing, you forgot `CONFIG_FILE`.
- Basic smoke test via Envoy (OpenAI-compatible):
  - Send a POST to `http://localhost:8801/v1/chat/completions` with `{"model":"auto", "messages":[{"role":"user","content":"hi"}]}` and check that the mock responds with `[mock-openai/gpt-oss-20b]` content when testing profile is active.

### DNS problems inside containers

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
