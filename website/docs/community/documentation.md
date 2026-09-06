# Documentation Guide

Public documentation lives under `website/`. Write for a reader trying to
understand or operate the system, not as a record of how a change was
implemented.

## Choose the right location

| Content | Location |
|---------|----------|
| Concepts, use cases, and architecture | `website/docs/overview/` |
| First run, configuration, deployment, and operations | The matching section in `website/sidebars.ts` |
| Signals, projections, decisions, algorithms, and plugins | `website/docs/tutorials/` |
| Stable HTTP or Kubernetes field reference | `website/docs/api/` |
| Contributor workflow | `website/docs/community/` or canonical repository contributor docs |

Avoid creating a second source of truth for generated schemas, config
inventories, or commands already owned by code. Link to the authoritative
reference or update its generator instead.

## Write for the task

A capability page should answer, in this order:

1. What problem does this solve?
2. When should a reader use it?
3. What is the smallest valid configuration or command?
4. What are its important limits, security implications, and dependencies?

Prefer one realistic example over several near-duplicates. Do not paste local
terminal transcripts, one-off test output, unqualified benchmark numbers, or
implementation scorecards into long-lived user documentation.

Use sentence-case headings, specify a language on fenced code blocks, and use
relative links for other docs pages. Put website images under
`website/static/img/`.

## Preview and validate

```bash
cd website
npm ci
npm run start
```

Before submitting:

```bash
cd website
npm test
npm run build:en
```

From the repository root, the docs-only CI path is:

```bash
make agent-docs-ci-gate AGENT_BASE_REF=origin/main
```

The build treats broken internal links as errors. Check external links that are
important to a procedure, especially downloads, charts, and upstream versioned
guides.

## Generated references

The configuration catalog is derived from `config/fragments/` and the first
sentence of each matching capability guide's **Overview**. Regenerate and check
it from the repository root:

```bash
make docs-config
make docs-config-check
```

The Operator field reference is generated from the current Go API types:

```bash
make docs-crd
make docs-crd-check
```

Edit the source fragment, capability guide, or Operator API comment rather than
editing a generated block by hand.

## Localization

English source pages live in `website/docs/`. Chinese translations live in:

```text
website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/
```

Keep translated paths aligned with their English source path. If a translation
cannot be updated in the same pull request, remove its current-version override
so Docusaurus serves the current English page. Keep historical `version-v*`
translations unchanged. `make docs-check-translations` treats that fallback as
coverage information while still failing on stale or invalid overrides.

For a new locale, add it to `website/docusaurus.config.ts`, generate the locale
catalog with `npm run write-translations -- --locale <locale>`, and validate a
locale-specific build.
