# Selection Package Notes

## Scope

- `src/semantic-router/pkg/selection/**`

## Responsibilities

- Keep runtime selection registry or factory wiring, algorithm-specific policy code, storage, metrics, and remote adapters on separate seams.
- Treat each algorithm family such as static, Elo, AutoMix, GMT, hybrid, and RL-driven selection as its own owner for policy-specific logic.
- Keep selector storage or telemetry support separate from the policy math that consumes it.

## Change Rules

- `selector.go`, `automix.go`, `hybrid.go`, `gmtrouter.go`, `rl_driven.go`, and `factory.go` are ratcheted hotspots. Add new policy branches, scoring helpers, or backend adapters in sibling modules instead of widening the existing files.
- Keep `factory.go` focused on construction and registration. New algorithm behavior should live with the algorithm, not in the factory.
- If a change touches storage, telemetry, and selection policy at once, split the support seam first instead of growing one of the hotspot policy files.
