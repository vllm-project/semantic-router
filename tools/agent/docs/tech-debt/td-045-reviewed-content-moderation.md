# TD045: Content Moderation Lacks a Reviewed Implementation

## Status

Open

## Owner Plan

[PL-0032: Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

## Release Relevance

Not release-blocking. This is a community-safety automation gap and must not
block runtime or artifact publication.

## Scope

GitHub issue, pull request, and comment moderation automation.

## Summary

The previous content moderation and cleanup workflows evaluated JavaScript
stored in `SPAM_DETECTION_SCRIPT`. The repository did not contain the
implementation, tests, review history, or a stable behavior contract. That
hidden-code path is disabled; automatic moderation remains off until a reviewed
implementation exists.

## Evidence

- `.github/workflows/anti-spam-filter.yml` loaded a repository secret and
  executed it with `eval` while holding issue, pull-request, and contents write
  permissions.
- `.github/workflows/cleanup-existing-spam.yml` wrapped and evaluated the same
  secret, then monkey-patched GitHub API methods to infer a detection result.
- No source-controlled detector or test corpus defines the intended moderation
  behavior, so replacing it with guessed heuristics would silently change
  policy.

## Why It Matters

Hidden executable code with write permissions cannot be reviewed, tested, or
reproduced. Guessing a replacement would either weaken moderation or create
unreviewed false-positive behavior.

## Desired End State

Use a reviewed, source-controlled detector with tests and an explicit policy,
or delegate moderation to a separately reviewed GitHub App. Keep read-only
analysis separate from mutation, require dry-run evidence before enabling
writes, and grant only issue/pull-request permissions to the mutation step.

## Exit Criteria

- Detection policy and implementation are source-controlled and reviewed.
- Representative clean and spam fixtures define expected behavior.
- Dry-run reporting is available without write permissions.
- Mutation is a separate reviewed step with minimal permissions.
- No workflow evaluates code from a secret.
