---
title: 文档指南
translation:
  source_commit: "f53d10fbf1021f9204e03afe2dd9a3374979e829"
  source_file: "docs/community/documentation.md"
  outdated: false
---

# 文档指南

公开文档在 `website/`。写给要理解或运维系统的读者，而不是记录某次改动怎么实现。

## 放对位置

| 内容 | 位置 |
|---------|----------|
| 概念、使用场景和架构 | `website/docs/overview/` |
| 首次运行、配置、部署和运维 | `website/sidebars.ts` 中对应小节 |
| 信号、投影、决策、算法和插件 | `website/docs/tutorials/` |
| 稳定的 HTTP 或 Kubernetes 字段参考 | `website/docs/api/` |
| 贡献者工作流 | `website/docs/community/` 或仓库内权威贡献文档 |

不要为代码已拥有的生成 schema、配置清单或命令再造一份真相源。链到权威参考，或更新它的生成器。

## 按任务写

能力页应按这个顺序回答：

1. 解决什么问题？
2. 读者何时该用它？
3. 最小可用配置或命令是什么？
4. 重要限制、安全含义和依赖是什么？

宁可用一个真实例子，也不要堆几个几乎重复的例子。不要把本地终端记录、一次性测试输出、未限定条件的基准数字或实现记分卡贴进长期用户文档。

标题用句式大小写，代码块标明语言，其他文档页用相对链接。网站图片放在 `website/static/img/`。

## 预览和校验

```bash
cd website
npm ci
npm run start
```

提交前：

```bash
cd website
npm test
npm run build:en
```

从仓库根目录跑与 CI 相同的变更文件路径：

```bash
make check BASE_REF=origin/main
```

构建会把内部断链当错误。对流程重要的外部链接也要检查，尤其是下载、图表和上游带版本的指南。

## 生成参考

配置目录由 `config/fragments/` 以及对应能力指南 **概览** 的第一句派生。从仓库根目录重新生成并检查：

```bash
make docs-config
make docs-config-check
```

Operator 字段参考由当前 Go API 类型生成：

```bash
make docs-crd
make docs-crd-check
```

改源 fragment、能力指南或 Operator API 注释，不要手改生成块。

## 本地化

英文源页在 `website/docs/`。中文译文在：

```text
website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/
```

译文路径要和英文源路径对齐。若同一 pull request 无法更新译文，就删掉当前版 override，让 Docusaurus 提供当前英文页。历史 `version-v*` 译文保持不动。`make docs-check-translations` 把这种回退当作覆盖信息，但仍会因过期或无效 override 失败。

新增语言时，把它加到 `website/docusaurus.config.ts`，用 `npm run write-translations -- --locale <locale>` 生成语言目录，并校验该语言构建。
