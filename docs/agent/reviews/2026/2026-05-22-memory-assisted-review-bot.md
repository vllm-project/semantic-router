# Review Brief

## Summary

Introduces a memory-assisted code review bot that classifies PRs by size, checks for an author-provided review brief, and optionally runs an AI-powered review using the brief as context. The safe rollout keeps deterministic classifier failures as the hard gate and treats AI review availability as advisory.

## Changed Areas

- `.github/workflows/` — Two new workflows: classifier (labels + comment) and AI review (context pack + inference + comment)
- `tools/agent/scripts/` — `agent_memory_classifier.py` (PR classification logic) and `agent_review_context.py` (bounded context pack builder)
- `tools/agent/scripts/tests/` — Unit tests cover classifier threshold/path/status handling, hard-gate behavior, workflow ordering, and review context generation.
- `docs/agent/reviews/` — Review brief format documentation and directory structure
- `tools/agent/skills/review-brief-authoring/` — Agent skill card for drafting briefs
- `.github/PULL_REQUEST_TEMPLATE.md` — Added Review Brief section
- `.prowlabels.yaml`, `tools/agent/repo-manifest.yaml`, `tools/agent/skill-registry.yaml`, `tools/agent/task-matrix.yaml` — Registry and manifest updates

## Key Decisions

- Threshold set at 500 changed lines (additions + deletions) to require a review brief
- Required or resolved review briefs must follow the documented path convention:
  `docs/agent/reviews/YYYY/YYYY-MM-DD-<kebab>.md`
- The PR template defaults to `Review brief: N/A`; authors replace it with a real review brief path only when required.
- The classifier hard gate is deterministic: brief requirement, path validity, changed-file status, existence, and non-empty content.
- AI review uses `openai/gpt-4.1` via `actions/ai-inference@v1` with 1200 max completion tokens
- Both workflows use `pull_request_target` to access label/comment write permissions while checking out base branch code only
- Missing required brief context is a hard failure after labels and comments are published
- AI review is advisory during the initial rollout; model unavailability or missing AI output does not block merge after the classifier hard gate passes.

## Validation

- Unit tests cover classifier threshold/path/status handling, hard-gate behavior, workflow ordering, and review context generation.
- YAML syntax validated for both workflow files
- Python syntax validated for both scripts

## Reviewer Focus

- Security model of `pull_request_target` + reading PR head content via API for AI inference input
- Prompt injection surface: PR body, review brief, and diff are injected into AI prompt with soft "treat as untrusted" instruction only
- Classifier threshold logic: whether 500 LOC (additions+deletions) is appropriate for this repo

## Risks or Follow-ups

- AI inference response is post-processed for a parseable brief/diff verdict, but AI unavailability is advisory during the initial rollout.
- No rate limiting or concurrency control on workflow triggers (repeated `edited` events)
- Prompt injection defense relies on system prompt instruction alone; no structural isolation or output sanitization
- `actions/ai-inference@v1` availability and quota are not validated in CI; the initial rollout avoids blocking merge when the model is unavailable.
- Future: parameterization needed for Marketplace publishing (model, threshold, path pattern)
