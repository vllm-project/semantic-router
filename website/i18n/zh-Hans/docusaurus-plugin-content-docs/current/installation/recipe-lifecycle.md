---
title: 托管配方生命周期
description: 用 vllm-sr CLI 对 Router 管理 API 规划、应用和删除配方。
translation:
  source_commit: "9649d02f2582471642196bde2e086c34ccb9c88c"
  source_file: "docs/installation/recipe-lifecycle.md"
  outdated: false
is_mtpe: true
---

# 托管配方生命周期 {#managed-recipe-lifecycle}

`vllm-sr recipe` 命令是 CLI 对 canonical Router 配置的有状态写入路径：它们读取配方文件、校验它，然后比较并交换进活跃配置。配方是什么、配方与入口和模型的关系，见[配方](../tutorials/global/recipes)。CLI 如何与其他配置界面配合，见[配置工作流](./configuration-workflows)。本页讲这些命令的生命周期。

操作活跃配置的配方命令通过 Router 管理 API 通信，并共享相同的连接选项：

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--endpoint` | 本地 Router API 端口 | Router 管理基 URL |
| `--token-env` | `VSR_MGMT_TOKEN` | 保存管理令牌的环境变量 |
| `--timeout` | `15` | 请求超时，秒 |

令牌从 `--token-env` 指定的环境变量读取；CLI 不接受把令牌写在命令行参数里。

## 检查 {#inspect}

```bash
vllm-sr recipe list
vllm-sr recipe get my-recipe
```

`list` 打印每个托管配方及集合 ETag，ETag 标识这次读取看到的状态。`get` 读取一个配方并在同一集合 ETag 下返回，因为 ETag 标识的是配置文档而非任何单个配方。下面的写命令运行时各自重新获取一个新集合 ETag。

## 校验 {#validate}

```bash
vllm-sr recipe validate recipe.yaml
```

校验根据配方契约检查文件，不触碰活跃配置。它和 `plan`、`apply` 运行时先行执行的校验是同一个。

## 规划 {#plan}

```bash
vllm-sr recipe plan recipe.yaml
```

`plan` 校验配方，把提议的动作连同当时的集合 ETag 一起报告。它不写任何东西。它的 ETag 只是参考：后续的 `apply` 会重新获取一个新 ETag，而不是要求 `plan` 打印的那个，所以规划不是锁，也不保证配置保持不变。用它审查提议的应用；如果当前配置对该决策重要，在应用前立即重跑一次。

## 应用 {#apply}

```bash
vllm-sr recipe apply recipe.yaml
```

`apply` 校验配方，然后用该命令开始时读取的集合 ETag 把它比较并交换进活跃配置。如果配置在该读取与写入之间发生变化，ETag 前置条件失败，什么都不会写入。较早 `plan` 之后、`apply` 开始之前发生的变化不会阻止应用；它仍可能对较新的状态写入成功。如果提议的变更需要重新审查，针对新状态重跑 `plan`。

## 删除 {#delete}

```bash
vllm-sr recipe delete my-recipe
```

删除前，先用 `get` 或 `list` 检查目标。`recipe plan recipe.yaml` 只预览一个应用提议；它不预览 `delete` 会移除什么。`delete` 获取一个新鲜的集合 ETag 并把一个配方比较并交换出配置，因此它只能防住与删除命令本身竞态的变更。只有未被引用的配方才能删除：只要还有入口指向它，删除即被拒绝（入口映射是唯一能引用配方的配置元素）。默认配方也不能删除——它是顶层路由画像，服务器以同样方式拒绝。删除只移除该配方，别无其他——没有级联，也没有撤销。

## 与 Dashboard 的交互 {#interaction-with-the-dashboard}

配方命令和 Dashboard 的配置编辑器通过不同路径作用于同一份 canonical 配置：命令经由 Router 管理 API，那里的每次写入在刚获取的集合 ETag 下比较并交换，而 Dashboard 直接写配置并自行传播到运行时。一个部署只用一个界面作为事实来源，另一个只用来检查，不要用它独立覆盖。ETag 前置条件只守护命令路径的写入；Dashboard 的写入不参与这套机制。
