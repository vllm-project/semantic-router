---
translation:
  source_commit: "bac2743"
  source_file: "docs/tutorials/observability/dashboard.md"
  outdated: true
---

# Semantic Router 仪表板 (Semantic Router Dashboard)

Semantic Router 仪表板是一个统一的操作员界面，集成了配置管理 (Configuration Management)、交互式演练场 (Interactive Playground) 以及实时监控与可观测性。它为本地开发、Docker Compose 和 Kubernetes 部署提供了一个统一的入口点。

- 统一查看和编辑配置（带有防护机制）
- 通过您喜欢的 UI (Open WebUI) 测试提示词
- 查看指标/仪表板 (Grafana/Prometheus)
- 统一后端代理，规范各服务之间的认证、CORS 和 CSP

## 包含内容

### 前端 (React + TypeScript + Vite)

现代化的单页应用 (SPA)，采用：

- React 18 + TypeScript + Vite
- React Router 实现客户端路由
- CSS Modules，支持持久化的深色/浅色主题
- 可折叠侧边栏，方便快速切换章节
- 由 React Flow 驱动的拓扑可视化

页面：

- 落地页：介绍和快速链接

![Dashboard Landing](/img/dashboard/landing.png)

- 演练场：内置聊天演练场，用于快速测试

- 配置：实时配置查看器/编辑器，具有结构化面板和原始视图

![Configuration Page](/img/dashboard/config.png)

- 拓扑：从用户请求到模型选择的视觉流程

![Topology View](/img/dashboard/topology.png)

- 监控：嵌入式 Grafana 仪表板

![Grafana Embedded](/img/dashboard/grafana.png)

### 后端 (Go HTTP 服务器)

- 提供前端构建产物（SPA 路由）
- 反向代理上游服务，并为 iframe 嵌入规范化 Header
- 为配置和工具数据库公开了一组简小的仪表板 API

关键路由：

- 健康检查：`GET /healthz`
- 配置（读）：`GET /api/router/config/all`（读取 YAML，返回 JSON）
- 配置（写）：`POST /api/router/config/update`（将 YAML 写回文件）
- 工具数据库：`GET /api/tools-db`（提供与配置同目录下的 tools_db.json）
- 路由 API：`GET/POST /api/router/*`（转发 Authorization Header）
- Grafana (嵌入)：`GET /embedded/grafana/*`
- Prometheus (嵌入)：`GET /embedded/prometheus/*`
- Open WebUI (嵌入)：`GET /embedded/openwebui/*`
- 路由指标透传：`GET /metrics/router` → 重定向到路由指标

代理会剥离/覆盖 `X-Frame-Options` 并调整 `Content-Security-Policy` 以允许 `frame-ancestors 'self'`，从而实现在仪表板同源下的安全嵌入。

## 环境变量

通过环境变量提供上游目标和运行时设置（括号内为默认值）：

- `DASHBOARD_PORT` (8700)
- `TARGET_GRAFANA_URL`
- `TARGET_PROMETHEUS_URL`
- `TARGET_ROUTER_API_URL` (http://localhost:8080)
- `TARGET_ROUTER_METRICS_URL` (http://localhost:9190/metrics)
- `TARGET_OPENWEBUI_URL` (可选)
- `ROUTER_CONFIG_PATH` (../../config/config.yaml)
- `DASHBOARD_STATIC_DIR` (../frontend)

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
