# Internationalization (i18n) Guide

This directory contains translations for the vLLM Semantic Router documentation.

## Directory Structure

```text
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
| ----- | ----------- |
| `source_commit` | Commit SHA of the English source file at translation time |
| `source_file` | Relative path to the English source file |
| `outdated` | **Auto-managed by CI** - do not manually edit |

## Workflow

### Auditing Translation Drift

Run the local audit before updating or reviewing translated docs:

```bash
make docs-check-translations
```

The default locale is `zh-Hans`. Override it when adding another locale:

```bash
make docs-check-translations DOCS_TRANSLATION_LOCALE=zh-Hans
```

The audit checks the English `website/docs/` tree against `i18n/{locale}/docusaurus-plugin-content-docs/current/` and reports:

| Status | Meaning |
| ------ | ------- |
| `Missing translations` | English docs that fall back to English in the translated locale |
| `Metadata issues` | Existing translated files missing `translation.source_commit`, `translation.source_file`, or `translation.outdated` |
| `Outdated translations` | English source files with commits after the recorded `source_commit` |
| `Status metadata mismatches` | `outdated` frontmatter does not match the computed source status |

The audit is read-only. It does not generate translations or update frontmatter. It returns nonzero when drift or metadata issues are found, so a failing `make docs-check-translations` can mean the check worked and found stale translations.

To update only the `translation.outdated` status flags, run:

```bash
make docs-fix-translation-status
```

This mode does not change translated prose or advance `source_commit`. It only makes the banner state honest while translators update the content. The make target succeeds when only translation drift remains and fails only for setup or usage errors.

Underlying script exit status guide:

| Exit | Meaning |
| ---- | ------- |
| `0` | All checked translations are synced and have valid metadata |
| `1` | The audit ran successfully and found missing, outdated, or metadata-incomplete translations |
| `2` | Usage or setup error, such as an unknown option, missing locale directory, or no usable `git` executable |

To inspect a report without failing a local shell session, append `|| true`:

```bash
make docs-check-translations || true
```

To verify a reported outdated file manually, compare the listed `source_commit` with the current English source history. For example:

```bash
git log --oneline 45bfd49e..HEAD -- website/docs/tutorials/projection/scores.md
```

The check currently covers only the `current` docs translation directory. Versioned docs under `version-v0.1`, `version-v0.2`, and `version-v0.3` are not audited by this script.

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

If the local audit reports `Status metadata mismatches`, update the translated content first when needed, then set `outdated: false` only after `source_commit` has been advanced to the reviewed English source commit.

Recommended order for a full zh-Hans sync:

1. Run `--fix-status` so pages with stale Chinese content show the outdated banner.
2. Add missing `translation` frontmatter to metadata-only files after checking which English source revision they match.
3. Translate missing files from `website/docs/` into `i18n/zh-Hans/docusaurus-plugin-content-docs/current/`.
4. Update outdated files by comparing `source_commit..HEAD` against each English source file.
5. For each reviewed translation, update `source_commit` to the latest English source commit and set `outdated: false`.
6. Run the audit again and build the Chinese site.

> **Note**: Do NOT manually change `outdated` field. CI will automatically set it to `false` after your PR is merged.

## CI Automation

A daily GitHub Action ([check-translation-staleness.yml](../../.github/workflows/check-translation-staleness.yml)) automatically manages the `outdated` status:

| Condition | Action |
| --------- | ------ |
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
