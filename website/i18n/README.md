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
is_mtpe: false                    # Set true only after human review
translation:
  source_commit: "abc1234def5678"  # Commit SHA when translation was made
  source_file: "docs/intro.md"     # Path to English source file
  outdated: false                  # Auto-updated by CI
---
```

### Fields

| Field | Description |
| ----- | ----------- |
| `is_mtpe` | Required top-level boolean: `true` records human review; `false` leaves the AI translation notice visible |
| `source_commit` | Commit SHA of the English source file at translation time |
| `source_file` | Relative path to the English source file |
| `outdated` | **Auto-managed by CI** - do not manually edit |

`is_mtpe` sits beside `translation`, not inside it. Use an unquoted `true` or
`false`; a quoted value is a string, and even `"false"` suppresses the notice in
the banner. The translation team decides when a page qualifies as reviewed.
The audit never assigns or changes this field. An outdated translation still
shows its warning regardless of `is_mtpe`.

## Workflow

### Auditing Translation Drift

Run the audit before updating or reviewing translated docs:

```bash
make docs-check-translations
```

The default locale is `zh-Hans`. Override it when adding another locale:

```bash
make docs-check-translations DOCS_TRANSLATION_LOCALE=zh-Hans
```

The audit uses two signals:

1. **Latest Git commit time (primary):** if the English file was committed after the translated file, the translation is definitely outdated. If the translated file is at least as new, it is considered likely synced.
2. **`translation.source_commit` (secondary):** if the recorded source commit is behind while the translated file is not older, the file is reported for metadata verification instead of being treated as definitely outdated.

| Status | Meaning |
| ------ | ------- |
| `Missing translations` | English docs that do not have a translated file |
| `Outdated translations` | English source has a newer latest commit than the translated file |
| `Metadata issues` | Translation metadata is missing, invalid, or points to the wrong source file |
| `Metadata needs verification` | The translated file is not older, but its recorded `source_commit` is behind |
| `Status metadata mismatches` | The `outdated` flag disagrees with an unambiguous timestamp result |

The audit is read-only and returns nonzero while translation or metadata work remains.

Review-state validation covers Markdown and MDX pages in both `current/` and
versioned translation directories. Missing or invalid `is_mtpe` declarations
appear under `Metadata issues`. Source-drift checks remain limited to current
documentation.

Current-document coverage is tracked as an exact, reviewable path inventory in
`translation-coverage-baseline.txt`. Deleting a recorded locale override fails
the audit while its English source still exists. New overrides and intentional
page moves or retirements require an explicit baseline refresh:

```bash
make docs-update-translation-baseline
```

Review the baseline diff with the documentation change. Versioned documentation
is not included in this inventory. The lightweight PR gate runs
`make docs-check-translation-coverage`, which checks this inventory without
turning existing source-drift or metadata work into a coverage failure.

To update only unambiguous `translation.outdated` flags, run:

```bash
make docs-fix-translation-status
```

This command does not translate prose, advance `source_commit`, or change
`is_mtpe`. It does not automatically change ambiguous `Metadata needs
verification` entries.

To inspect the report without failing a local shell session:

```bash
make docs-check-translations || true
```

### Adding a New Translation

1. Copy the English file to `i18n/{locale}/docusaurus-plugin-content-docs/current/`
2. Get current commit SHA of the source file:

   ```bash
   git log -1 --format="%h" -- website/docs/your-file.md
   ```

3. Add the frontmatter with that commit SHA and an explicit `is_mtpe` boolean
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

A weekly maintenance job
([maintenance.yml](../../.github/workflows/maintenance.yml), using the reusable
`check-translation-staleness.yml` workflow) automatically manages the
`outdated` status:

| Condition | Action |
| --------- | ------ |
| English latest commit is newer than translation | Sets `outdated: true` |
| Translation is not older and `source_commit` is current | Sets `outdated: false` |
| Translation is not older but `source_commit` is behind | Reports for verification without changing status |
| No changes needed | Skips PR creation |

### How It Works

1. Runs daily at UTC 00:00
2. Scans all locales under `website/i18n/*/`
3. Runs the same timestamp-primary and metadata-secondary check used locally
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
