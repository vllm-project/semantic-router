# 贡献指南

:::info[权威来源]
本页内容生成自 [CONTRIBUTING.md](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)，以代码仓库中的文件为准。
:::

## 快速开始

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
make download-models  # 从 HuggingFace 下载 ML 模型
make build            # 构建 Rust + Go 组件
make test             # 运行所有测试
```

## 工作流程

### 创建分支

```bash
git checkout -b feature/your-feature-name
```

### 本地构建和测试

```bash
make clean && make build && make test
```

### 运行端到端测试

```bash
make run-envoy &
make run-router &
python e2e/testing/run_all_tests.py
```

### 运行 Pre-commit 检查

```bash
pre-commit run --all-files
```

如未安装：`pip install pre-commit && pre-commit install`

### 使用 DCO 签名提交

所有提交**必须**签名（DCO 要求）。不使用 `-s` 参数，CI 将**拒绝**你的 PR。

```bash
git commit -s -m "feat: add something"
```

### 提交 PR

- 目标分支：`main`
- 包含：描述 + 关联 issue 链接 + 测试结果
- `make test` 和端到端测试必须通过

## 调试

| 组件 | 调试方法 |
|------|----------|
| Envoy | 查看 `make run-envoy` 终端的请求/响应日志 |
| Router | 查看 `make run-router` 终端的路由决策 |
| Rust | `RUST_LOG=debug`（级别：trace/debug/info/warn/error） |
| Go | `SR_LOG_LEVEL=debug` |

## 相关指南

- **[开发指南](./development)**：环境要求、构建、运行测试
- **[文档指南](./documentation)**：如何编写和翻译文档
- **[AI 翻译指南](./translation-guide)**：AI 翻译 Prompt 和术语表
- **[代码规范](./code-style)**：格式化、lint、pre-commit hooks
