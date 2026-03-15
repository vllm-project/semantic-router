---
translation:
  source_commit: "1d0aec2"
  source_file: "docs/tutorials/observability/dashboard.md"
  outdated: true
---

# Semantic Router 仪表板 (Semantic Router Dashboard)

Semantic Router 仪表板结合了落地页/初始化流程与一个带认证的操作控制平面。它现在覆盖配置生命周期、在线测试、可观测性、调试、评测以及 agent 运维，并在本地开发、Docker Compose 和 Kubernetes 部署之间提供统一入口。

- 提供落地页、登录与 setup 启动流程，便于首次引导
- 统一承载 dashboard、监控、链路追踪、状态、日志、回放、配置、演练场与拓扑等操作面
- 覆盖评测、评分、builder 与 ML 模型选择工作流
- 提供 OpenClaw 与 MCP 能力，用于 agent 工作区、服务端与工具执行
- 统一后端代理，规范 Grafana、Prometheus、Jaeger 与路由 API 的认证、CORS 和 CSP

## 包含内容

### 前端 (React + TypeScript + Vite)

现代化的单页应用 (SPA)，采用：

- React 18 + TypeScript + Vite
- React Router 实现客户端路由
- CSS Modules，支持持久化的深色/浅色主题
- 可折叠侧边栏，方便快速切换章节
- 由 React Flow 驱动的拓扑可视化

主要界面：

- 入口流程：`/`、`/login`、`/auth/transition`、`/setup`
- 操作首页：`/dashboard`、`/monitoring`
- 配置与测试：`/config`、`/config/:section`、`/playground`、`/playground/fullscreen`、`/topology`
- 调试与运行时洞察：`/tracing`、`/status`、`/logs`、`/replay`
- 质量与优化：`/evaluation`、`/ml-setup`、`/ratings`、`/builder`
- Agent 与管理界面：`/clawos`、`/users`

### 后端 (Go HTTP 服务器)

- 提供前端构建产物与受认证保护的 SPA 路由
- 反向代理 Grafana、Prometheus、Jaeger、Router API 以及部分集成服务 API
- 提供 setup、配置生命周期、工具、评测、ML pipeline、MCP 与 OpenClaw API

关键路由：

- 健康与初始化：`/healthz`、`/api/settings`、`/api/setup/state`、`/api/setup/import-remote`、`/api/setup/validate`、`/api/setup/activate`
- 认证与管理：`/api/auth/*`、`/api/admin/users`、`/api/admin/permissions`、`/api/admin/audit-logs`
- 配置生命周期：`/api/router/config/all`、`/api/router/config/yaml`、`/api/router/config/update`、`/api/router/config/deploy/preview`、`/api/router/config/deploy`、`/api/router/config/rollback`、`/api/router/config/versions`、`/api/router/config/defaults`、`/api/router/config/defaults/update`
- 工具与运维 API：`/api/tools-db`、`/api/tools/web-search`、`/api/tools/open-web`、`/api/tools/fetch-raw`、`/api/status`、`/api/logs`、`/api/topology/test-query`
- 评测与 ML 工作流：`/api/evaluation/*`、`/api/ml-pipeline/*`
- Agent 集成：`/api/mcp/*`、`/api/openclaw/*`、`/embedded/grafana/*`、`/embedded/prometheus/*`、`/embedded/jaeger*`、`/metrics/router`

代理会剥离/覆盖 `X-Frame-Options` 并调整 `Content-Security-Policy` 以允许 `frame-ancestors 'self'`，从而实现在仪表板同源下的安全嵌入。

## 环境变量

常用环境变量：

- `DASHBOARD_PORT` (8700)
- `TARGET_GRAFANA_URL`
- `TARGET_PROMETHEUS_URL`
- `TARGET_JAEGER_URL`
- `TARGET_ROUTER_API_URL` (http://localhost:8080)
- `TARGET_ROUTER_METRICS_URL` (http://localhost:9190/metrics)
- `TARGET_ENVOY_URL`（Playground 通过 Envoy 聊天时需要）
- `ROUTER_CONFIG_PATH` (../../config/config.yaml)
- `DASHBOARD_STATIC_DIR` (../frontend)
- `DASHBOARD_READONLY`
- `DASHBOARD_SETUP_MODE`
- `ML_SERVICE_URL`
- `DASHBOARD_AUTH_DB_PATH`、`DASHBOARD_JWT_SECRET`、`DASHBOARD_ADMIN_EMAIL`、`DASHBOARD_ADMIN_PASSWORD`、`DASHBOARD_ADMIN_NAME`
- `OPENCLAW_ENABLED`、`OPENCLAW_URL`、`OPENCLAW_DATA_DIR`、`OPENCLAW_TOKEN`

注意：配置更新 API 会写入 `ROUTER_CONFIG_PATH`。在容器/Kubernetes 中，此路径必须是可写的（不能是只读的 ConfigMap）。如果您需要持久化运行时编辑，请挂载一个可写卷。

## 快速开始

### Docker Compose (推荐)

仪表板已集成到主 Compose 文件中。

```bash
# 从仓库根目录执行
make docker-compose-up
```

然后在浏览器中打开：

- 仪表板：http://localhost:8700
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## 相关文档

- [安装配置](../../installation/configuration.md)
- [可观测性指标](./metrics.md)
- [分布式链路追踪](./distributed-tracing.md)
