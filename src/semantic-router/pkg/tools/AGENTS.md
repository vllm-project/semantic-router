# Tools Package Notes

## Scope

- `src/semantic-router/pkg/tools/**`

## Responsibilities

- Keep tool registry, request or response tool policy, and relevance scoring on separate seams.
- Treat `relevance.go` as the narrow scoring seam and `tools.go` as the higher-level orchestration hotspot.
- Keep transport or provider-specific behavior out of the shared tool-policy layer unless a dedicated adapter owns it.

## Change Rules

- `tools.go` is a ratcheted hotspot. New relevance heuristics, registry helpers, or policy branches should be extracted into sibling modules instead of widening the main orchestration file.
- Do not mix prompt parsing, tool execution policy, and scoring math in one new helper.
- If a change only affects scoring or ranking, prefer `relevance.go` or a new adjacent helper rather than growing `tools.go`.
