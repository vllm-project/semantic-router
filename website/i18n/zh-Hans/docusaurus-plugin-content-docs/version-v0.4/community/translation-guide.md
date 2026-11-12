---
title: 翻译指南
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/community/translation-guide.md"
  outdated: false
---

# 翻译指南

翻译页面含义，不要照搬英文句式。命令、配置键、API 路径、代码标识和产品名必须与原文完全一致。

## 翻译检查清单

- 从 `website/docs/` 下的当前英文页开始。
- 保留 front matter、标题层级、代码块、提示框、链接和图片路径。
- 对 signal、projection、decision、algorithm、plugin、recipe、model card 等路由概念使用同一套译法。
- 不要翻译 YAML 键、环境变量、请求头名、CLI 标志、文件名，或属于线协议的取值。
- 重新跑语言构建，确认标题不会破坏锚点或导航。

当前版中文页镜像英文树，位于：

```text
website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/
```

只有译文仍然准确时才保留语言 override。若译文跟不上英文源，就删掉当前版 override；Docusaurus 会显示当前英文页，而不是发布过期译文。该回退只适用于当前文档。不要删除 `version-v*` 下的历史译文。

在 `website/` 下校验语言：

```bash
npm run build:zh
```

从仓库根目录审计已有译文，并报告英文回退覆盖：

```bash
make docs-check-translations
```

机翻可以当草稿，但发布前必须由贡献者审术语、技术含义和所有可执行示例。
