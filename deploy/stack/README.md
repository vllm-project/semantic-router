# Stack Deployment

Single-container deployment for vLLM Semantic Router with all components bundled together.

## Components

| Service    | Port  | Description                 |
| ---------- | ----- | --------------------------- |
| Envoy      | 8801  | API Gateway (main endpoint) |
| Dashboard  | 8700  | Web UI for monitoring       |
| Grafana    | 3000  | Metrics dashboards          |
| Prometheus | 9090  | Metrics collection          |
| Jaeger     | 16686 | Distributed tracing         |
| LLM-Katan  | 8002  | Lightweight LLM server      |

## Quick Start

```bash
# Build
docker build -f tools/docker/Dockerfile.stack -t vsr-stack:latest .

# Download models (required)
make download-models

# Run
docker run -d --name vsr \
  -p 8801:8801 -p 8002:8002 -p 8700:8700 -p 3000:3000 -p 9090:9090 \
  -p 16686:16686 \
  -v $(pwd)/models:/app/models:ro \
  vsr-stack:latest
```

## Files

| File                           | Description                   |
| ------------------------------ | ----------------------------- |
| `config.stack.yaml`            | Semantic router configuration |
| `envoy.template.yaml`          | Envoy proxy configuration     |
| `supervisord.stack.conf`       | Process management            |
| `entrypoint-stack.sh`          | Container entrypoint          |
| `prometheus.yaml`              | Prometheus scrape config      |
| `grafana-*.yaml`               | Grafana provisioning          |

## Environment Variables

Key variables that can be overridden:

```bash
ENVOY_LISTEN_PORT=8801
DASHBOARD_PORT=8700
LLMKATAN_MODEL=/app/models/Qwen/Qwen3-0.6B
GF_SECURITY_ADMIN_PASSWORD=admin
```

## Services Management

```bash
# Enter container
docker exec -it vsr bash

# View all service status
supervisorctl status

# View specific service logs
tail -f /var/log/supervisor/semantic-router.log
tail -f /var/log/supervisor/envoy.log

# Restart a service
supervisorctl restart semantic-router
supervisorctl restart envoy

# Restart all services
supervisorctl restart all
```

## Notes

- Image size is ~10GB
- Requires models to be mounted at `/app/models`
- All services managed by Supervisor
