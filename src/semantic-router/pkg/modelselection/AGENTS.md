# Model Selection Package Notes

## Scope

- `src/semantic-router/pkg/modelselection/**`

## Responsibilities

- Keep runtime selector behavior, offline analysis, persistence, and benchmark-running support on separate seams.
- Treat benchmark and trainer utilities as offline tooling, not as a reason to widen the runtime selector path.
- Keep config analysis and persistence helpers decoupled from scoring or selection orchestration.

## Change Rules

- `selector.go` and `benchmark_runner.go` are ratcheted hotspots. New benchmark reporting, fixture loading, or selector branches belong in adjacent helpers instead of widening those files.
- Do not mix training-data preparation or offline benchmarking with request-time routing behavior in the same code path.
- Keep persistence helpers narrow and reusable instead of turning `selector.go` into a second storage owner.
