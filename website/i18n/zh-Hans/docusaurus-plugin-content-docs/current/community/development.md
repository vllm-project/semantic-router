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
vllm-sr serve --image-pull-policy never
```

该构建会安装可编辑的 `vllm-sr` CLI，并创建本地 Router、控制面板和 Envoy 镜像。`--image-pull-policy never` 确保运行使用这些本地镜像。

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
vllm-sr serve --image-pull-policy never --platform amd
```

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
