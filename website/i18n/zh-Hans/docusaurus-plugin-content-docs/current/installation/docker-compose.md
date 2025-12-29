---
translation:
  source_commit: "8e37ee0"
  source_file: "docs/installation/docker-compose.md"
  outdated: false
sidebar_position: 3
---

# 使用 Docker Compose 安装

本指南提供了使用 Docker Compose 部署带有 Envoy AI Gateway 的 vLLM Semantic Router 的分步说明。

## 通用前提条件

- **Docker Engine:** 更多信息请参阅 [Docker Engine 安装](https://docs.docker.com/engine/install/)

- **克隆仓库:**

  ```bash
  git clone https://github.com/vllm-project/semantic-router.git
  cd semantic-router
  ```

- **下载分类模型（约 1.5GB，仅首次运行需要）:**

  ```bash
  # 提示：如果遇到错误 'hf: command not found'，请运行 'pip install huggingface_hub hf_transfer'
  make download-models
  ```

  这将下载 Router 使用的分类模型：

  - Category classifier（ModernBERT-base）
  - PII classifier（ModernBERT-base）
  - Jailbreak classifier（ModernBERT-base）

---

### 要求

- Docker Compose v2（`docker compose` 命令，而不是旧版的 `docker-compose`）

  安装 Docker Compose 插件（如果缺失），更多信息请参阅 [Docker Compose 插件安装](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

  ```bash
  # 对于 Debian / Ubuntu
  sudo apt-get update
  sudo apt-get install -y docker-compose-plugin

  # 对于 RHEL / CentOS / Fedora
  sudo yum update -y
  sudo yum install -y docker-compose-plugin

  # 验证
  docker compose version
  ```

- 确保端口 8801、50051、19000、3000 和 9090 空闲

### 启动服务

```bash
# 核心（router + envoy）
docker compose -f deploy/docker-compose/docker-compose.yml up --build

# 后台运行（确认无误后推荐）
docker compose -f deploy/docker-compose/docker-compose.yml up -d --build

# 包含 mock vLLM + testing profile（将 Router 指向 mock endpoint）
CONFIG_FILE=/app/config/testing/config.testing.yaml \
  docker compose -f deploy/docker-compose/docker-compose.yml --profile testing up --build
```

### 验证

- gRPC: `localhost:50051`
- Envoy HTTP: `http://localhost:8801`
- Envoy Admin: `http://localhost:19000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`（首次登录使用 `admin` / `admin`）

### 常用操作

```bash
# 查看服务状态
docker compose ps

# 跟踪 Router 服务的日志
docker compose logs -f semantic-router

# 进入 Router 容器
docker compose exec semantic-router bash

# 更改配置后重新创建
docker compose -f deploy/docker-compose/docker-compose.yml up -d --build

# 停止并清理容器
docker compose down
```
