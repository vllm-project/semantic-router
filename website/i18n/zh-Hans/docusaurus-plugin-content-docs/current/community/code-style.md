---
title: 代码风格与质量
translation:
  source_commit: "f53d10fbf1021f9204e03afe2dd9a3374979e829"
  source_file: "docs/community/code-style.md"
  outdated: false
---

# 代码风格与质量

格式化和静态检查由仓库配置强制执行。使用已入库的工具，不要另维护一套编辑器规则。

## 运行共享检查

安装仓库管理的 hook，并对已跟踪文件运行：

```bash
make precommit-install
make precommit-check
```

复现容器化 pre-commit 工作流：

```bash
make precommit-local
```

常规变更文件路径：

```bash
make impact ENV=cpu CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
```

## 语言约定

### Go

- 用 `gofmt` 格式化。
- 优先保持包内聚，只有在所有权或可测性变好时才拆分。
- 导出 API 的用途不明显时再写文档。
- 用 `make check-go-mod-tidy` 校验模块元数据。
- 用 `make go-lint` 跑仓库的 lint 配置。

### Rust

- 用 `cargo fmt` 格式化。
- 通过仓库报告的校验路径跑 `cargo clippy`。
- 正常失败路径返回带类型的错误，不要 panic。
- 把 unsafe 和 FFI 边界做小，并写清楚。

### Python

- 支持你正在改的包所声明的 Python 版本。
- 在公开和非平凡接口上使用类型标注。
- 把命令编排和可复用逻辑分开。
- 组件有 Make 目标时，用该目标跑测试。

### TypeScript 和 React

- 遵循控制面板的 ESLint 和 TypeScript 配置。
- 当 helper 或 hook 负责数据获取和转换时，不要把这些逻辑放进展示组件。
- 为用户可见行为补充聚焦的组件或 E2E 覆盖。

## 生成文件

不要手改生成的 API 参考、schema 或目录块。改源头并跑所属生成器，然后在同一 pull request 里同时提交源和输出。
