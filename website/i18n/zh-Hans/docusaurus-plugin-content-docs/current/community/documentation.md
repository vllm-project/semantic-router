# 文档指南

本指南介绍如何为 vLLM Semantic Router 贡献文档。

## 目录结构

文档使用 Docusaurus 构建。

- `website/docs/`：主要英文文档（Markdown）。
- `website/i18n/`：本地化文档（例如 `zh-Hans` 代表中文）。
- `website/docusaurus.config.ts`：站点配置。
- `website/sidebars.ts`：侧边栏导航。

## 编辑文档

1. **定位文件：** 在 `website/docs/` 中找到 Markdown 文件。
2. **进行更改：** 使用 Markdown 语法编辑内容。
3. **本地预览：**

   ```bash
   cd website
   npm run start
   ```

4. **验证链接：** 确保所有相对链接正确。
5. **语法检查：** 运行 `make markdown-lint` 进行语法检查。

## 国际化 (i18n)

我们支持多种语言（如英语、中文）。默认语言为英语。

### 添加新页面

1. 在 `website/docs/` 中创建英文文件。
2. 在 `website/i18n/{locale}/docusaurus-plugin-content-docs/current/` 中创建对应的翻译文件。
   - 中文示例：`website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/`。
3. 确保文件名和目录结构完全匹配。

### 添加新语言

1. 在 `website/docusaurus.config.ts` 中配置新语言环境 (locale)。
2. 运行 `npm run write-translations -- --locale <new-locale>` 生成 JSON 翻译文件。
3. 将 `docs` 目录结构复制到 `website/i18n/<new-locale>/...` 并翻译 Markdown 文件。

### 更新翻译

更新英文文档时，请尽可能同时更新中文翻译。如果你无法翻译，请开启 Issue 寻求帮助。

### 利用 LLM 加速翻译流程

你可以参考我们的 [AI 自动翻译指南](./translation-guide) 来使用 LLM 进行辅助翻译。该指南包含了推荐的 Prompt 和术语表，能显著提高翻译效率和一致性。

## 风格指南

- **标题：** 使用句首大写（Sentence case）。
- **代码块：** 指定语言（例如 \`\`\`bash）。
- **链接：** 内部链接使用相对路径。
- **图片：** 将图片放置在 `website/static/img/` 并使用 `/img/...` 引用。
