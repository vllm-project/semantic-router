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
| CLI 命令、HTTP API 或 Kubernetes 字段参考 | `website/docs/api/` |
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
python3 -m pip install -r requirements.txt
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

源文件变更应同时提交对应的生成产物。检查命令会重新计算预期内容并与已提交的文件比较，发现过期即失败，不会改写文件；网站构建也不会自动修复过期参考。

| 参考 | 权威源 | 重新生成 | 检查 |
| --- | --- | --- | --- |
| Model Hub 与内置 catalog | `config/catalog/` 和内置 recipe bundles | `make model-catalog-generate` | `make model-catalog-generated-check` |
| [CLI 命令](../api/cli) | `src/vllm-sr/cli/` 中注册的 Click 命令 | `make docs-cli` | `make docs-cli-check` |
| 配置目录 | `config/fragments/` 与能力指南的 **Overview** | `make docs-config` | `make docs-config-check` |
| OpenAPI 与端点索引 | Go API 路由目录与配置 schema | `make api-docs-generate` | `make api-docs-check` |
| Operator 字段参考 | Operator Go API 类型及注释 | `make docs-crd` | `make docs-crd-check` |
| 公开运维 skill | `tools/agent/skills/vllm-sr-agent-operations/` | `make agent-skill-sync` | `make agent-skill-check` |
| GitHub 社区统计 | 生成器、身份信息与带日期的 GitHub 快照 | 在 `website/` 运行 `npm run contributors:rank` / `npm run committers:activity` | `make docs-community-check` |

`make docs-generated-check` 检查 catalog、CLI 参考、配置目录与公开 skill，不依赖原生库构建。每次 PR 和 main 校验都执行该检查，包括只改源文件或文档的情况。`npm run build` 和 `npm test` 也会先执行这些参考检查。它们需要 Python 3.10 或更新版本以及 `website/requirements.txt` 中的依赖；可通过 `VLLM_SR_DOCS_PYTHON` 指定 Python 解释器。

`make generated-contract-check` 还会检查配置 schema、OpenAPI 与 Operator 参考，使用现有 Go/原生库构建前置条件。`make generated-contract-generate` 按依赖顺序刷新这些公开参考。应修改源文件或生成器，不要手改生成块。

社区统计是外部动态数据的带日期快照。离线检查同时验证生成源和生成内容的摘要，不与持续变化的 GitHub 活跃度比较。修改生成器或身份信息后必须成功刷新；网络失败不能静默复用源文件已过期的快照。

网站在构建时导入模型 catalog。重新生成并提交后，还需要成功发布生产网站。排查网站过期时，可将线上 `/model-catalog/catalog.json` 与目标版本的 `website/static/model-catalog/catalog.json` 比较；Git 内文件一致并不代表该版本已经上线。

## 本地化

英文源页在 `website/docs/`。中文译文在：

```text
website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/
```

译文路径要和英文源路径对齐。若同一 pull request 无法更新译文，就删掉当前版 override，让 Docusaurus 提供当前英文页。历史 `version-v*` 译文保持不动。`make docs-check-translations` 把这种回退当作覆盖信息，但仍会因过期或无效 override 失败。

新增语言时，把它加到 `website/docusaurus.config.ts`，用 `npm run write-translations -- --locale <locale>` 生成语言目录，并校验该语言构建。
