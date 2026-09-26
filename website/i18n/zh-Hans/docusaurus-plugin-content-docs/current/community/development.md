---
title: 开发指南
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/community/development.md"
  outdated: false
---

# 开发指南

影响 Router 或 CLI 行为的改动，请使用仓库的本地镜像流程。它构建的是贡献者在本地验证时使用的同一套服务拓扑。

## 前置条件

- Git
- GNU Make
- Docker 或 Podman
- Python 3.10 或更高版本，用于 CLI、测试、训练和模拟器工具

仓库引导目标会创建 Python 环境，并安装校验工具链：

```bash
make harness-bootstrap
```

个别子项目可能还有额外要求。不要安装仓库根目录的 `requirements.txt`，它不存在。使用你正在改的组件旁边的依赖文件或包元数据。

## 本地构建和运行

```bash
make vllm-sr-dev
VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest \
  vllm-sr serve --image-pull-policy never
```

该构建会安装可编辑的 `vllm-sr` CLI，构建标记为 `latest` 的 Router 和控制面板镜像，并确保官方 Envoy 镜像可用。
即使采用可编辑安装，只要包版本号是稳定版本，CLI 默认仍会选择对应的发布镜像，因此需要显式设置 `VLLM_SR_IMAGE`。
CLI 会推导出相同 tag 的官方控制面板镜像；`--image-pull-policy never` 则禁止拉取缺失的镜像。

常用生命周期命令：

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

ROCm 相关工作：

```bash
make vllm-sr-dev VLLM_SR_PLATFORM=amd
VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest \
  vllm-sr serve --image-pull-policy never --platform amd
```

如果自定义了 `DOCKER_TAG`、`DOCKER_REGISTRY` 或 Make 的镜像变量，请通过 `VLLM_SR_IMAGE` 将实际构建的镜像传给 `serve`，必要时同时设置 `VLLM_SR_DASHBOARD_IMAGE`。
构建完成后的提示会打印包含所选镜像的启动命令。

## 选择正确的测试

先看仓库事实，再跑所属域的检查：

```bash
make impact ENV=cpu CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
```

常见定向套件包括：

```bash
# Router 和原生绑定
make test-semantic-router
make test-binding

# 分类器
make test-category-classifier
make test-pii-classifier
make test-jailbreak-classifier

# Python CLI
make vllm-sr-test

# 机队模拟器
make vllm-sr-sim-test
```

当变更会通过启动、路由、API、部署配置或其他在线路经表现出来时，显式选择集成或 E2E：

```bash
make verify DOMAIN=<domain>
make verify PROFILE=<profile>
```

## 测试后端

从源码运行 provider mocker 需要 Python 3.11 或更高版本。统一的 mocker 为协议、路由和故障测试提供确定性响应，在同一个轻量服务中覆盖 OpenAI Chat Completions、Responses、Anthropic Messages 和图像测试数据：

```bash
make test-provider-mocker
make docker-run-provider-mocker
# 也可以在独立 Python 环境中直接运行：
make start-provider-mocker
```

设置 `PROVIDER_MOCKER_IMAGE` 可以复用已有镜像；未设置时，Docker 目标在本地构建服务。mocker 独立于产品 release 维护。镜像标签对应运行时代码、依赖锁文件、Dockerfile 和 `.dockerignore` 的内容哈希，仅文档或测试改动会复用已有镜像。CI 将标签解析为镜像 digest，并让各测试任务使用同一个产物；只有上述构建输入变化时才发布新的辅助镜像。

需要真实生成时，使用可选的 tiny-model runner。它统一运行 `Qwen/Qwen3-0.6B`，固定上游 llama.cpp CPU 镜像 digest、模型 revision 和校验和，将 Q8_0 权重下载到忽略的缓存目录，并关闭 thinking，不再构建额外的推理镜像：

```bash
make tiny-model-smoke  # 健康检查、真实文本、SSE 结束和停止字符串
make tiny-model-serve # 在 localhost:8000 前台运行真实后端
```

模型 smoke 限制 CPU、内存、上下文和输出长度。测试 Router 行为时，在独立终端运行后端，通过 `vllm-sr serve` 转发请求；协议边界条件由确定性测试覆盖。

## 校验本地栈

已配置的 listener 是面向客户端的端点。空工作区生成的设置是 `http://localhost:8899`：

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

使用当前配置中的虚拟模型名。使用自定义 listener 或端口偏移时，`vllm-sr status` 会显示栈和已发布端口。

## 调试

- 先用 `vllm-sr logs <service>` 看组件日志，再依赖容器名。
- 原生库诊断设 `RUST_LOG=debug`。
- Router 诊断设 `SR_LOG_LEVEL=debug`。
- 在运行时调试配置前，先跑 `vllm-sr config validate --config <file>`。
- 启动和网络失败见[常见错误](/zh-Hans/docs/troubleshooting/common-errors)和[容器连通性](/zh-Hans/docs/troubleshooting/container-connectivity)。
