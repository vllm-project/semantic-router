---
sidebar_position: 3
---

# Containerized Deployment

This unified guide helps you quickly run Semantic Router locally (Docker Compose) or in a cluster (Kubernetes) and explains when to choose each path.Both share the same configuration concepts: **Docker Compose** is ideal for rapid iteration and demos, while **Kubernetes** is suited for long‑running workloads, elasticity, and upcoming Operator / CRD scenarios.

## Choosing a Path

**Docker Compose path** = semantic-router + Envoy proxy + optional mock vLLM (testing profile) + Prometheus + Grafana. It gives you an end-to-end local playground with minimal friction.

**Kubernetes path** (current manifests, MVP) = semantic-router Deployment (gRPC + metrics), a PVC for model cache, its ConfigMap, two Services (gRPC + metrics), **an Envoy gateway Deployment + Service (NodePort by default)**. It still does **NOT** bundle: real LLM inference backends, Prometheus/Grafana stack, Istio / Gateway API / Operator / CRDs.

| Scenario / Goal                             | Recommended Path                 | Why                                                                                     |
| ------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------- |
| Local dev, quickest iteration, hacking code | Docker Compose                   | One command starts router + Envoy + (optionally) mock vLLM + observability stack        |
| Demo with dashboard quickly                 | Docker Compose (testing profile) | Bundled Prometheus + Grafana + mock responses                                           |
| Team shared staging / pre‑prod              | Kubernetes                       | Declarative config, rolling upgrades, persistent model volume, now with gateway (Envoy) |
| Performance, scalability, autoscaling       | Kubernetes                       | HPA, scheduling, resource isolation                                                     |
| Future Operator / CRD driven config         | Kubernetes                       | Native controller pattern                                                               |
| Progressive hardening / production gateway  | Kubernetes                       | Can iteratively add TLS, mTLS, Service Mesh, Observability, traffic policies            |

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
- _Optional: Prometheus metrics stack (e.g. [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator))_
- _(Planned / not yet merged) Service Mesh or advanced gateway:_
  - _[Istio](https://istio.io/latest/docs/setup/getting-started/) / [Kubernetes Gateway API](https://gateway-api.sigs.k8s.io/)_
- Separate deployment of **Envoy** (or another gateway) + real **LLM endpoints** (follow [Installation guide](https://vllm-semantic-router.com/docs/getting-started/installation)).
  - Replace placeholder IPs in `deploy/kubernetes/config.yaml` once services exist.

### What the Kubernetes manifests now include (MVP)

After the Envoy MVP addition the base kustomize set provides:

| Component                            | Purpose                                             |
| ------------------------------------ | --------------------------------------------------- |
| `semantic-router` Deployment         | gRPC ExtProc intelligence layer                     |
| Init container (Python)              | One‑time model download into PVC                    |
| PVC + mount                          | Persistent model cache across pod restarts          |
| ConfigMap (`semantic-router-config`) | Router configuration + tools DB                     |
| Service `semantic-router`            | ClusterIP gRPC (50051)                              |
| Service `semantic-router-metrics`    | ClusterIP metrics (9190)                            |
| **Envoy Deployment**                 | HTTP ingress + ExtProc bridge (8801, 19000 admin)   |
| **Envoy Service (NodePort)**         | External access (defaults: 30801→8801, 30900→19000) |

### Still NOT included

| Missing Piece                                      | Add Later For                                                   |
| -------------------------------------------------- | --------------------------------------------------------------- |
| Real LLM backends (vLLM / Ollama / custom)         | Actual completions (currently you must deploy separately)       |
| Prometheus / Grafana stack                         | Metrics scraping & dashboards (import existing JSON once added) |
| Service Mesh / Gateway API / Istio                 | Advanced traffic policies, mTLS, canary, auth                   |
| TLS termination / Ingress                          | Production-grade external exposure                              |
| External / persistent semantic cache (Redis, etc.) | Cross‑pod cache sharing & warm retention                        |
| Health checks for downstream LLM clusters          | Automatic endpoint eviction, circuit breaking                   |
| HorizontalPodAutoscaler                            | Elastic scaling based on QPS / CPU / custom metrics             |

### Deploy (Kustomize)

```bash
kubectl apply -k deploy/kubernetes/

# Wait for pod
kubectl -n semantic-router get pods
```

Manifests create (recap):

- Deployment (semantic-router + init model downloader)
- Envoy Deployment (gateway)
- Service `semantic-router` (gRPC 50051)
- Service `semantic-router-metrics` (metrics 9190)
- Envoy Service (NodePort 8801/19000 externally)
- ConfigMap (router config + tools DB)
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

| Goal                    | Change                                              |
| ----------------------- | --------------------------------------------------- |
| Scale horizontally      | `kubectl scale deploy/semantic-router --replicas=N` |
| Resource tuning         | Edit `resources:` in `deployment.yaml`              |
| Add HTTP readiness      | Switch TCP probe -> HTTP `/health` (port 8080)      |
| PVC size                | Adjust storage request in PVC manifest              |
| Metrics scraping        | Add ServiceMonitor / scrape rule                    |
| Expose via LoadBalancer | Change Envoy Service `type: LoadBalancer`           |
| Use Ingress + TLS       | Set Envoy Service to ClusterIP + create Ingress     |
| Change Envoy log level  | Edit args in `envoy-deployment.yaml`                |

---

## Envoy Configuration Tuning (Kubernetes Path)

The Envoy MVP lives under `deploy/kubernetes/envoy/` and uses a minimal ORIGINAL_DST pattern to route to dynamic LLM endpoints whose IP:Port is injected by the router through the `x-semantic-destination-endpoint` header. Key knobs:

| Aspect                 | Where                                                                                          | Notes                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| External exposure      | `envoy-service.yaml`                                                                           | Switch `type: NodePort` → `LoadBalancer` for cloud; or use Ingress in front.                 |
| Access log verbosity   | Deployment `args`                                                                              | Adjust `--component-log-level ext_proc:trace,http:info` for lower noise.                     |
| Timeouts               | `envoy-config.yaml`                                                                            | `connect_timeout`, `message_timeout`, route `timeout`—tune for longest model latency.        |
| ORIGINAL_DST header    | `envoy-config.yaml`                                                                            | Must remain synced with router’s configured header name (`x-semantic-destination-endpoint`). |
| Buffer sizes           | `per_connection_buffer_limit_bytes`                                                            | Increase if large prompt bodies cause upstream resets.                                       |
| Health checks (future) | Add `health_checks` per cluster when you move from ORIGINAL_DST to static clusters.            | None                                                                                         |
| Static model clusters  | Replace ORIGINAL_DST cluster with per‑model STRICT_DNS and route on header `x-selected-model`. | None                                                                                         |

### When to move beyond ORIGINAL_DST

Move to explicit static (or STRICT_DNS) clusters if you need: per‑model circuit breaking, health checking, weighted load balancing, or to hide raw IPs. The router would then emit logical model headers instead of raw endpoint IP:Port.

---

## Connecting Real LLM Backends

The Kubernetes manifests **do not** deploy any LLM inference servers. You must bring (or reference) your own. Three common patterns:

### 1. In‑Cluster vLLM Deployment (recommended for testing)

Create a simple vLLM Deployment & Service:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-phi4
  namespace: semantic-router
spec:
  replicas: 1
  selector:
    matchLabels: { app: vllm-phi4 }
  template:
    metadata:
      labels: { app: vllm-phi4 }
    spec:
      containers:
        - name: vllm
          image: your-registry/vllm:latest
          args:
            [
              "serve",
              "microsoft/phi-4",
              "--port",
              "8000",
              "--served-model-name",
              "phi4",
            ]
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-phi4
  namespace: semantic-router
spec:
  selector:
    app: vllm-phi4
  ports:
    - port: 8000
      targetPort: 8000
```

Get its ClusterIP:

```bash
kubectl -n semantic-router get svc vllm-phi4 -o jsonpath='{.spec.clusterIP}'
```

Use that IP in `deploy/kubernetes/config.yaml` under `vllm_endpoints.address` (the router currently requires an IP literal). Apply & restart:

```bash
kubectl apply -k deploy/kubernetes/
kubectl -n semantic-router rollout restart deploy/semantic-router
```

### 2. External (Managed) Endpoint

If your LLM runs outside the cluster (e.g. on another VM or managed service):

1. Ensure the node network can reach the external IP:Port (firewall / VPC rules).
2. Put the public/private IP into `config.yaml` (`address` must be IP, no DNS/hostname).
3. If egress is restricted, add NetworkPolicy exceptions.
4. Consider latency: add higher route timeout in Envoy if needed.

### 3. Transition to Static Envoy Clusters (Optional Advanced)

Instead of passing raw endpoint IP:Port through `x-semantic-destination-endpoint`, you can define per‑model clusters in Envoy and have the router set a logical header (e.g. `x-selected-model`). Route config then chooses clusters by header match. Benefits: health checks, circuit breaking, weighted load balancing.

---

## Quick Routing Verification (Kubernetes)

Once a backend is wired:

```bash
curl -X POST http://<NodeIP or LB>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
```

Check Envoy logs for `selected_model` & `selected_endpoint` fields and semantic-router logs for the classification decision.

---

## Updating LLM Endpoints

1. Edit `deploy/kubernetes/config.yaml` (add/remove endpoint blocks).
2. Re-apply kustomize.
3. Restart only the router (Envoy usually does not need a restart with ORIGINAL_DST):

```bash
kubectl apply -k deploy/kubernetes/
kubectl -n semantic-router rollout restart deploy/semantic-router
```

If you migrate to static clusters, Envoy must also be reloaded (Deployment restart) when clusters change.

---

## Hardening & Next Steps

| Step                            | Rationale                               |
| ------------------------------- | --------------------------------------- |
| Add HPA for router & Envoy      | Scale under load                        |
| Introduce Prometheus Operator   | Automated metrics scraping              |
| Import dashboard JSON           | Visualize routing & latency             |
| Add NetworkPolicies             | Limit lateral movement / egress         |
| Enable TLS (Ingress / Gateway)  | Secure external traffic                 |
| External semantic cache (Redis) | Cross-pod cache hit ratio improvement   |
| Circuit breaking & retries      | Graceful handling of flaky LLM backends |
| Move to logical clusters        | Health checks & traffic shaping         |

---

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
  - Send a POST (Docker: `http://localhost:8801/...`; K8s: `http://<NodeIP or LB>:8801/...`). With the MVP K8s path you must have added a real LLM backend for a non-error response.
  - If `x-semantic-destination-endpoint` points to `127.0.0.1` in K8s logs, you forgot to replace placeholder IPs with a reachable Service ClusterIP or external IP.

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
