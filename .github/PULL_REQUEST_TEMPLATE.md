<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION BELOW AND CONFIRM THE CHECKLIST ITEMS.

Closes #xxxx

## Purpose

- What does this PR change?
- Why is this change needed?
- Which module(s) does this affect? `Router` / `CLI` / `Dashboard` / `Operator` / `Fleet-Sim` / `Bindings` / `Training` / `E2E` / `Docs` / `CI/Build`

## Test Plan

- What commands, checks, or manual steps should reviewers use?
- Why is this validation sufficient for the affected module(s)?

## Test Result

- What were the actual results?
- Any follow-up risks, gaps, or blockers?

## Review Brief

Review brief: N/A

<!-- If required, replace N/A with:
docs/agent/reviews/YYYY/YYYY-MM-DD-<short-kebab-title>.md
-->

- Required for large PRs when requested by the agent memory classifier.
- The PR author confirms the brief matches the code, changed files, PR body, and validation evidence.

---
<details>
<summary>Semantic Router PR Checklist</summary>

- [ ] PR title uses module-aligned prefixes such as `[Router]`, `[CLI]`, `[Dashboard]`, `[Operator]`, `[Fleet-Sim]`, `[Bindings]`, `[Training]`, `[E2E]`, `[Docs]`, or `[CI/Build]`
- [ ] If the PR spans multiple modules, the title includes all relevant prefixes
- [ ] Commits in this PR are signed off with `git commit -s`
- [ ] The Purpose, Test Plan, and Test Result sections reflect the actual scope, commands, and blockers for this change
- [ ] If a review brief is provided, it is author-confirmed and does not include secrets, private data, raw logs, or chain-of-thought

</details>

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full contributor workflow and commit guidance.
