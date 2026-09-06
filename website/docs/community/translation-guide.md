# Translation Guide

Translate the meaning of a page, not its English sentence structure. Commands,
configuration keys, API paths, code identifiers, and product names must remain
exactly as they appear in the source.

## Translation checklist

- Start from the current English page under `website/docs/`.
- Preserve front matter, heading levels, code fences, admonitions, links, and
  image paths.
- Use one consistent translation for routing concepts such as signal,
  projection, decision, algorithm, plugin, recipe, and model card.
- Do not translate YAML keys, environment variables, header names, CLI flags,
  filenames, or values that are part of a wire contract.
- Re-run the locale build and check that headings do not break anchors or
  navigation.

Chinese current-version pages mirror the English tree under:

```text
website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/
```

Only keep a locale override while it is accurate. If a translated page cannot
be updated with its English source, remove the current-version override;
Docusaurus will show the current English page instead of publishing a stale
translation. This fallback applies only to the current docs. Do not delete
historical translations under `version-v*`.

Validate the locale from `website/`:

```bash
npm run build:zh
```

From the repository root, audit every translation that is present and report
English fallback coverage with:

```bash
make docs-check-translations
```

Machine translation can be a draft, but a contributor must review terminology,
technical meaning, and all executable examples before the translation is
published.
