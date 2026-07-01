---
name: review-brief-authoring
category: support
description: Use when drafting or updating an author-provided review brief from the current PR diff, changed files, PR body, and validation evidence.
---

# Review Brief Authoring

## Trigger

- A PR needs an author-provided review brief under `docs/agent/reviews/YYYY/`
- A large or risky change needs compact reviewer context for humans and review automation
- The agent memory classifier labels a PR as `agent-memory-missing`

## Required Surfaces

- `harness_docs`
- `contributor_interface`

## Stop Conditions

- The PR is below the memory threshold (currently 500 changed lines) and no brief is explicitly requested
- A review brief already exists and the PR author has confirmed it matches the code
- The PR is docs-only or website-only with no behavior-visible changes
- Insufficient evidence to ground the brief (no diff, no test output, no PR body)

## Workflow

1. Inspect the current PR diff, changed files, PR body, and available test output.
2. Determine the brief path using the naming convention:
   `docs/agent/reviews/YYYY/YYYY-MM-DD-<short-kebab-title>.md`
3. Draft the brief using the exact template from `docs/agent/reviews/README.md`:
   - Summary, Changed Areas, Key Decisions, Validation, Reviewer Focus, Risks or Follow-ups
4. Mark claims that are not directly supported by the available evidence as `Not verified` or `Author stated`.
5. Keep plans, risks, and incomplete work in `Risks or Follow-ups`.
6. Add the brief path reference to the PR body under the Review Brief section.
7. Ask the PR author to confirm the brief matches the code before relying on it for review.

## Must Read

- [docs/agent/reviews/README.md](../../../../docs/agent/reviews/README.md)
- [.github/PULL_REQUEST_TEMPLATE.md](../../../../.github/PULL_REQUEST_TEMPLATE.md)
- [.prowlabels.yaml](../../../../.prowlabels.yaml)

## Standard Commands

- `git diff --stat`
- `git diff`
- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `python3 tools/agent/scripts/agent_memory_classifier.py --metadata <file> --output <file>`

## Gotchas

- Do not draft a brief from author intent or chat description alone.
- Do not invent test results or convert planned validation into completed validation.
- Do not include secrets, credentials, private data, raw logs, or chain-of-thought.
- Treat the brief as author-provided context; the diff and actual validation evidence remain authoritative.
- The brief path must pass the classifier regex: year directory must match the date prefix in the filename.
- Do not describe planned work as completed work; put incomplete items in Risks or Follow-ups.

## Acceptance

- The brief follows the repository template and path convention.
- The brief path passes validation by `agent_memory_classifier.py`.
- The brief is grounded in the PR diff, changed files, PR body, and available validation evidence.
- Unverified claims are explicitly marked `Not verified` or `Author stated`.
- The PR author owns final confirmation that the brief and code agree.
