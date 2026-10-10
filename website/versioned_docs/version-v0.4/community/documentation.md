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
| CLI commands, HTTP APIs, or Kubernetes fields | `website/docs/api/` |
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
python3 -m pip install -r requirements.txt
npm ci
npm run start
```

Before submitting:

```bash
cd website
npm test
npm run build:en
```

From the repository root, run the same changed-file path as CI:

```bash
make check BASE_REF=origin/main
```

The build treats broken internal links as errors. Check external links that are
important to a procedure, especially downloads, charts, and upstream versioned
guides.

## Generated references

Commit generated references together with the source change. Check commands
compare the committed output with freshly rendered content and fail without
rewriting it; running a website build does not repair stale references.

| Reference | Authoritative source | Regenerate | Check |
| --- | --- | --- | --- |
| Model Hub and built-in catalog | `config/catalog/` and built-in recipe bundles | `make model-catalog-generate` | `make model-catalog-generated-check` |
| [CLI commands](../api/cli) | Registered Click commands in `src/vllm-sr/cli/` | `make docs-cli` | `make docs-cli-check` |
| Configuration catalog | `config/fragments/` and capability guides' **Overview** | `make docs-config` | `make docs-config-check` |
| OpenAPI and endpoint index | Go API route catalog and config schema | `make api-docs-generate` | `make api-docs-check` |
| Operator field reference | Operator Go API types and comments | `make docs-crd` | `make docs-crd-check` |
| Public operations skill | `tools/agent/skills/vllm-sr-agent-operations/` | `make agent-skill-sync` | `make agent-skill-check` |
| GitHub community statistics | Generator sources, identity inputs, and the dated GitHub snapshot | `npm run contributors:rank` / `npm run committers:activity` in `website/` | `make docs-community-check` |

`make docs-generated-check` checks the catalog, CLI reference, configuration
catalog, and public skill without building native libraries. Every PR and main
validation runs this gate, including changes confined to source files or docs.
`npm run build` and `npm test` run the same reference checks first. They require
Python 3.10 or newer and the dependencies in `website/requirements.txt`; set
`VLLM_SR_DOCS_PYTHON` to choose a Python interpreter explicitly.

Community statistics are dated snapshots of changing external data. Their
offline gate verifies both the generating source digest and the generated body
digest, rather than comparing against live GitHub activity. Changing a generator
or its identity inputs requires a successful refresh; a network failure cannot
silently reuse a snapshot from obsolete sources.

Normal development, build, and deploy commands use the committed community
snapshots. Refresh them only with the explicit commands in the table above.

`make generated-contract-check` also checks the config schema, OpenAPI, and
Operator reference, using the normal Go/native build prerequisites.
`make generated-contract-generate` refreshes those public references in
dependency order. Edit the owning source or generator instead of hand-editing
generated output.

The website imports its model catalog at build time. Regenerating and committing
the catalog must be followed by a successful production website deployment.
To diagnose an outdated site, compare `/model-catalog/catalog.json` from the
deployed site with `website/static/model-catalog/catalog.json` in the intended
revision; matching source files in Git alone do not prove that revision is live.

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
