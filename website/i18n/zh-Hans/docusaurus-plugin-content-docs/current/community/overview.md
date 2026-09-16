---
title: 贡献
translation:
  source_commit: "867155c924b6527d6a412e1412ce712a9e5cc9b8"
  source_file: "docs/community/overview.md"
  outdated: false
---

# 贡献

欢迎向 Router、CLI、控制面板、部署资产、文档和评测工具贡献。仓库的
[CONTRIBUTING.md](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)
是权威工作流；本页只给出多数贡献者需要的检查路径。

## 开始之前

- 开新工作前先搜索已有 issue 和 pull request，避免重复。
- 改带本地规则的模块前，先读最近的 `AGENTS.md`。
- 一个 pull request 只聚焦一个行为或文档结果。
- 变更影响可观察行为时，补充或更新测试。

## 本地工作流

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router

make harness-bootstrap
make impact ENV=cpu CHANGED_FILES="path/to/changed-file"
```

`impact` 会报告所有权、最低检查，以及变更路径对应的候选 CI 或 E2E。它不会替你选开发流程。只有 ROCm 相关行为才用 `ENV=amd`。

默认本地镜像流程：

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never
```

针对性测试和运行时命令见[开发指南](./development)。

## 提交 pull request 之前

运行你这次变更对应的检查。范围匹配时，这些仓库级入口很有用：

```bash
make check CHANGED_FILES="path/to/changed-file"
make ci-full  # 高风险变更或完整本地 PR 对齐
```

pull request 中的每个提交都必须带 Developer Certificate of Origin 签名：

```bash
git commit -s -m "describe the change"
```

在 pull request 里说明问题、用户可见结果，以及你跑过的验证。只有帮助评审视觉变更时才附截图。

## 贡献者指南

- [模型与提供商 Day-0 支持](./model-provider-day-0-support)
- [开发指南](./development)
- [文档指南](./documentation)
- [代码风格与质量](./code-style)
- [架构概览](/zh-Hans/docs/overview/semantic-router-overview)
