# Looper Package Notes

## Scope

- `src/semantic-router/pkg/looper/**`

## Responsibilities

- Keep looper client transport, confidence scoring, ratings, RL-driven policy, and shared base types on separate seams.
- Treat the package as routing-policy runtime code, not as a home for unrelated config or response-pipeline behavior.
- Keep token filtering, telemetry, and storage or state helpers out of the main policy files when a narrower helper can own them.

## Change Rules

- `base.go`, `client.go`, `confidence.go`, `ratings.go`, and `rl_driven.go` are ratcheted hotspots. New evaluators, client helpers, or RL policy branches should be extracted into sibling helpers first.
- Do not mix outbound client behavior, local scoring math, and persistence logic in the same new code path.
- If a change touches both looper request transport and selector policy, treat that as a design smell and look for a narrower seam.
