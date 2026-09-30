---
name: maintainer-ops
description: Use for vLLM Semantic Router issue intake, PR decisions, releases, and reviewed GitHub mutations performed on behalf of maintainers.
---

# Maintainer operations

Read `tools/agent/docs/maintainer-ops.md` and
`tools/agent/maintainer-policy.yaml`. Resolve the exact issue, PR, release, and
head revision before mutating GitHub. Base decisions on current `main`, the
actual diff, tests, CI, and related history—not on descriptions alone.

Keep discovery read-only. Store local state only under
`.agent-harness/maintainer/`; acquire the documented resource lease before an
apply step and release it afterward. Prefer editing a mistaken comment over
posting a duplicate. Never expose private infrastructure in public artifacts.

For PRs, put blockers first and distinguish suggestions. For issue acceptance,
use the exact `/accept` comment rather than applying the label directly. For a
release, verify the immutable commit and artifacts before publication.
