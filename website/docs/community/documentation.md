# Documentation Guide

This guide covers how to contribute to the vLLM Semantic Router documentation.

## Directory Structure

The documentation is built using Docusaurus.

- `website/docs/`: Main English documentation (Markdown).
- `website/i18n/`: Localized documentation (e.g., `zh-Hans` for Chinese).
- `website/docusaurus.config.ts`: Site configuration.
- `website/sidebars.ts`: Sidebar navigation.
- `website/scripts/check-doc-structure.py`: Reachability and locale-structure checks for public docs.

## Editing Documentation

1. **Locate the file:** Find the Markdown file in `website/docs/`.
2. **Make changes:** Edit the content using Markdown syntax.
3. **Keep navigation aligned:** Every public English page in `website/docs/` must be reachable from `website/sidebars.ts` unless it is intentionally excluded by policy and validation.
4. **Preview locally:**

   ```bash
   make docs-dev
   ```

5. **Run structure and build checks:**

   ```bash
   make docs-check-structure
   make docs-build
   ```

6. **Lint website code when needed:**

   ```bash
   make docs-lint
   ```

## Internationalization (i18n)

We support multiple languages (e.g., English, Chinese). The default language is English.

### Adding a New Page

1. **Create the English file** in `website/docs/`.
2. **Create the corresponding translated file** in `website/i18n/{locale}/docusaurus-plugin-content-docs/current/`.
   - Example for Chinese: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/`.
3. **Ensure the filename and directory structure match exactly**, unless the locale uses a documented exception that is explicitly allowlisted by `check-doc-structure.py`.

### Adding a New Language

1. **Configure the new locale** in `website/docusaurus.config.ts`.
2. **Run** `npm run write-translations -- --locale <new-locale>` to generate JSON translation files.
3. **Copy the `docs` folder structure** to `website/i18n/<new-locale>/...` and translate the Markdown files.

### Updating Translations

When you update an English document, please also update the Chinese translation if possible. If you cannot translate it, please open an issue to request help.

Use the translation sync helper to inspect staleness:

```bash
./website/scripts/check-translation-sync.sh --locale zh-Hans
```

## Style Guide

- **Headings:** Use sentence case.
- **Code Blocks:** Specify the language (e.g., \`\`\`bash).
- **Links:** Use relative paths for internal links.
- **Images:** Place images in `website/static/img/` and reference them as `/img/...`.
