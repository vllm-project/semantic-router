# Internationalization (i18n) Guide

This directory contains translations for the vLLM Semantic Router documentation.

## Directory Structure

```
i18n/
└── {locale}/                              # e.g., zh-Hans
    ├── code.json                          # General UI translations
    ├── docusaurus-theme-classic/
    │   ├── navbar.json                    # Navigation bar labels
    │   └── footer.json                    # Footer labels
    └── docusaurus-plugin-content-docs/
        └── current/                       # Translations for /docs (next version)
```

## Translation Frontmatter

Each translated markdown file must include translation metadata:

```yaml
---
translation:
  source_commit: "abc1234def5678"  # Commit SHA when translation was made
  source_file: "docs/intro.md"     # Path to English source file
  outdated: false                  # Auto-updated by CI
---
```

### Fields

| Field | Description |
|-------|-------------|
| `source_commit` | Commit SHA of the English source file at translation time |
| `source_file` | Relative path to the English source file |
| `outdated` | **Auto-managed by CI** - do not manually edit |

## Workflow

### Adding a New Translation

1. Copy the English file to `i18n/{locale}/docusaurus-plugin-content-docs/current/`
2. Get current commit SHA of the source file:

   ```bash
   git log -1 --format="%h" -- website/docs/your-file.md
   ```

3. Add the frontmatter with that commit SHA
4. Translate the content
5. Submit a PR

### Updating an Outdated Translation

1. Check files marked with `outdated: true`
2. Compare with English source to identify changes
3. Update the translation content
4. Update `source_commit` to the latest commit of the English source:

   ```bash
   git log -1 --format="%h" -- website/docs/your-file.md
   ```

5. Submit a PR

> **Note**: Do NOT manually change `outdated` field. CI will automatically set it to `false` after your PR is merged.

## CI Automation

A daily GitHub Action ([check-translation-staleness.yml](../.github/workflows/check-translation-staleness.yml)) automatically manages the `outdated` status:

| Condition | Action |
|-----------|--------|
| `source_commit` is behind English source | Sets `outdated: true` |
| `source_commit` matches English source | Sets `outdated: false` |
| No changes needed | Skips PR creation |

### How It Works

1. Runs daily at UTC 00:00
2. Scans all locales under `website/i18n/*/`
3. Compares each translation's `source_commit` with current source file
4. Creates a single PR with all status updates (if any changes)
5. Closes any existing stale PR before creating a new one

## TranslationBanner Component

Import the banner in your translated MDX files to display status:

```jsx
import TranslationBanner from '@site/src/components/TranslationBanner';

<TranslationBanner />
```

Displays:

- 🤖 Blue: "This page was translated from English"
- ⚠️ Yellow: "This translation may be outdated" (when `outdated: true`)
