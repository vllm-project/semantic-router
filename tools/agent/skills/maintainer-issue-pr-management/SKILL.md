---
name: maintainer-issue-pr-management
category: support
description: Manages GitHub issue and pull-request lifecycle including creation, updates, triage labelling, and closeout metadata using canonical templates and repository taxonomy. Use when a maintainer asks to create, update, close, or triage GitHub issues or PRs, or when issue creation requires codebase analysis for scope, labels, or acceptance criteria.
---

# Maintainer Issue And PR Management

## Trigger

- Use when a maintainer asks the agent to create, update, close, or otherwise manage a GitHub issue or PR
- Use when issue creation needs codebase analysis before deciding scope, ownership, labels, acceptance criteria, or validation

## Required Surfaces

- `contributor_interface`

## Conditional Surfaces

- `harness_docs`
- `harness_exec`
- `router_config_contract`
- `signal_runtime`
- `decision_logic`
- `algorithm_selection`
- `plugin_runtime`
- `python_cli_schema`
- `python_cli_runtime`
- `dashboard_config_ui`
- `dashboard_console_backend`
- `k8s_operator`
- `training_post_training`

## Stop Conditions

- The requested issue or PR action depends on code intent that has not been inspected in the current repository state
- The requested labels, title, or validation summary would contradict the canonical templates or current harness rules
- A destructive delete is requested without a clear target and rationale; prefer close with reason unless the maintainer explicitly wants deletion

## Must Read

- [AGENTS.md](../../../../AGENTS.md)
- [docs/agent/README.md](../../../../docs/agent/README.md)
- [docs/agent/governance.md](../../../../docs/agent/governance.md)
- [CONTRIBUTING.md](../../../../CONTRIBUTING.md)
- [.github/PULL_REQUEST_TEMPLATE.md](../../../../.github/PULL_REQUEST_TEMPLATE.md)
- [.github/ISSUE_TEMPLATE/001_feature_request.yaml](../../../../.github/ISSUE_TEMPLATE/001_feature_request.yaml)
- [.github/ISSUE_TEMPLATE/002_bug_report.yaml](../../../../.github/ISSUE_TEMPLATE/002_bug_report.yaml)
- [.prowlabels.yaml](../../../../.prowlabels.yaml)

## Workflow

1. Resolve the repo context first.
   - If the issue or PR is based on a code change, inspect the relevant code with `codebase-retrieval` before drafting anything.
   - If changed paths are known or can be estimated, run `make agent-report ENV=cpu|amd CHANGED_FILES="..."` to resolve the primary skill, impacted surfaces, and expected validation.
2. Classify the artifact before writing.
   - Issues use the canonical issue templates as the schema for title and body.
   - PRs use the canonical PR template as the schema for title, summary, impacted surfaces, skipped surfaces, debt entry, and validation fields.
3. Apply canonical labels and naming.
   - For issues, start from the template defaults such as `bug` or `feature request`.
   - Add maintainer labels from `.prowlabels.yaml`; today that taxonomy includes `area` labels and `priority` labels.
   - Do not invent labels outside the repository taxonomy unless the maintainer explicitly asks for a new label.
4. Keep code analysis and management metadata aligned.
   - New implementation issues should cite the relevant files, symbols, or surfaces discovered during analysis.
   - If the work spans multiple resumable loops, link or update the indexed execution plan instead of hiding the plan only in the issue body.
   - If the desired architecture still diverges from the repo after the planned change, link or update the indexed tech-debt entry instead of leaving the gap only in the issue or PR text.
5. Enforce PR conventions.
   - PR titles must use the classified prefixes from `.github/PULL_REQUEST_TEMPLATE.md`, adding multiple prefixes when the change spans categories.
   - Commits intended for PRs must use `git commit -s`.
   - Commit messages do not need to repeat the PR title prefixes unless the maintainer explicitly wants them.

## Gotchas

- Do not draft labels, validation summaries, or scope from memory; inspect the current templates, labels, and repo state first.
- If an issue or PR exposes real architecture debt or multi-loop execution needs, put that state in the indexed debt or plan docs instead of hiding it only in the ticket text.

## Standard Commands

- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-validate`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`

## Acceptance

- Issue and PR management requests are grounded in the current repo state rather than memory
- Issue drafts include the canonical title/body shape plus the correct default and maintainer-applied labels
- PR drafts or updates include the canonical title classification, signoff expectation, summary fields, and validation results or blockers
