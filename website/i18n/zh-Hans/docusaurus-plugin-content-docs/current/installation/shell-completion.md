---
title: Shell 补全
description: 在 bash、zsh 和 fish 中为 vllm-sr CLI 安装 tab 补全。
translation:
  source_commit: "9649d02f2582471642196bde2e086c34ccb9c88c"
  source_file: "docs/installation/shell-completion.md"
  outdated: false
is_mtpe: true
---

# Shell 补全 {#shell-completion}

`vllm-sr completion` 为 CLI 生成并安装 tab 补全。补全覆盖命令名、子命令和选项——包括 `recipe` 和 `benchmark` 命令组——这样无需记住标志名就能驱动 CLI。

补全由 CLI 自身生成，并按 shell 安装。等 CLI 在 `PATH` 上之后再设置；本节的安装指南讲了怎么把 CLI 装上 PATH。

## 安装 {#install}

```bash
vllm-sr completion install
```

不带参数时，shell 从 `SHELL` 环境变量探测。传入 shell 名可显式为某一个安装：

```bash
vllm-sr completion install zsh
```

受支持的 shell 有 `bash`、`zsh` 和 `fish`。安装做什么取决于 shell：

| Shell | 动作 | 位置 |
|---|---|---|
| bash | 追加一行 eval | `~/.bashrc` |
| zsh | 追加一行 eval | `~/.zshrc` |
| fish | 写入补全脚本 | `~/.config/fish/completions/vllm-sr.fish` |

bash 和 zsh 条目在 shell 启动时求值 `vllm-sr completion show`，因此它们自动跟随 CLI 升级：每个新 shell 都从安装的 CLI 版本构建补全，升级后无需任何操作。fish 条目是写一次的静态文件——升级 CLI 后重跑 `vllm-sr completion install fish` 刷新它。

安装可安全重复。bash 和 zsh 的安装会向 rc 文件写入 `# vllm-sr shell completion` 标记，标记已存在时跳过，因此重复运行不会追加重复行。

## 改为打印脚本 {#print-the-script-instead}

要自己接线补全，或检查一次安装会添加什么，打印脚本并用你自己的工具处理：

```bash
vllm-sr completion show fish
```

shell 探测规则同样适用：不带参数的 `vllm-sr completion show` 从 `SHELL` 解析 shell。

## 故障排查 {#troubleshooting}

`Could not detect shell. Please specify one of: bash, zsh, fish`——`SHELL` 的值不匹配受支持的 shell。显式传入 shell：`vllm-sr completion install bash`。

安装后补全不出现——打开一个新 shell，或重新 source rc 文件（`source ~/.bashrc` 或 `source ~/.zshrc`）。对 fish，补全在下次 shell 启动时从 completions 目录加载。
