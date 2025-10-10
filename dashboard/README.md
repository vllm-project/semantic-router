# Modern Dashboard for Semantic Router

This dashboard provides a unified entry point for configuration management, interactive playground, and real-time monitoring & observability.

## Structure

- `frontend/`: UI for configuration, playground, monitoring
    - Monitoring (iframe Grafana + Prometheus charts)
    - Config Viewer (Read-only VSR config JSON)
    - Playground (iframe Open WebUI)
- `backend/`: API proxy, authentication, metrics aggregation
    - `/api/metrics` → Prometheus
    - `/api/config`  → semantic-router API / config loader
    - `/api/proxy`   → Open WebUI proxy
- `deploy/`: Deployment setups
    - `docker/`: Docker Compose setup
    - `kubernetes/`: K8s manifests
    - `local/`: Local/dev setup
- `helm-chart/`: (optional) Helm chart for dashboard

## Key Features

- **Grafana Integration**: Embedded dashboards via iframe, with reverse proxy and authentication.
- **Deployment Flexibility**: Local, Docker Compose, and Kubernetes support.
- **Security & Access Control**: Centralized authentication and role-based access control.
- **Extensibility**: Modular structure for future integrations.

## Getting Started

1. Choose your deployment method (`local/`, `docker/`, or `kubernetes/`).
2. Follow the setup instructions in the respective folder.
3. Access the dashboard via the provided URL.

## Future Extensions
- Alerts, advanced analytics, third-party integrations, Frontend UI, Backend API etc.

---
For more details, see individual folder READMEs and configuration files.