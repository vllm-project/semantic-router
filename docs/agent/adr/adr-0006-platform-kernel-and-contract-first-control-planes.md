# ADR 0006: Converge on a Platform Kernel With Contract-First Control Planes

## Status

Accepted

## Context

The 2026-04 full-repository modular architecture audit found that the repository's top-level shape is coherent, but its cross-stack boundaries are not. The repo already behaves like a multi-language platform monorepo with a router kernel plus several edge control planes:

- `src/semantic-router/` owns the runtime kernel, routing policy, and runtime-owned adapters.
- `src/vllm-sr/` owns the Python CLI and local runtime orchestration.
- `dashboard/backend/` and `dashboard/frontend/` own the human-facing control plane.
- `deploy/operator/` owns the Kubernetes-facing control plane.

The problem is not that these surfaces exist. The problem is that the control planes still reach too far into router-runtime internals and co-own implementation-level contract knowledge:

- dashboard backend imports router config and authoring internals directly instead of consuming a narrow service or contract seam
- operator controllers translate CRD state into router canonical config types directly
- the CLI still carries a large share of canonical contract knowledge and its local hotspot rules had drifted behind the real orchestration files
- runtime bootstrap and shared service publication still depend on process-wide globals rather than an explicit runtime graph

Without a durable architecture decision, the repo will keep tracking these issues as local file hotspots without converging on one cross-stack dependency rule.

## Decision

Adopt the following repository architecture target.

- Treat the repository as a platform kernel with contract-first control planes.
- Keep router policy and runtime internals owned by `src/semantic-router/` and avoid making those internals the default shared dependency for other products.
- Require cross-stack control planes such as the CLI, dashboard backend, and operator to consume versioned contracts or explicit public runtime-service seams instead of deep router-runtime internals wherever the dependency crosses a product boundary.
- Treat shared config and transport contracts as first-class seams. Schema/version ownership, runtime loading, and control-plane translation must not collapse back into one shared package boundary.
- Treat UI stores and route shells as interface surfaces, not deployment-orchestration owners. Frontend code may initiate control-plane actions, but durable deploy/apply/status semantics belong behind backend-owned service seams.
- Treat `src/vllm-sr/cli/core.py` and `src/vllm-sr/cli/commands/runtime.py` as the active CLI orchestration hotspots. Keep `docker_cli.py` as a thin compatibility or import-stability layer instead of the default place for new runtime flow.
- Use indexed technical debt entries for unresolved subsystem gaps and indexed execution plans for multi-loop convergence work, but keep this ADR as the durable top-level dependency direction.

## Consequences

- Subsystem refactors should now be judged against one explicit target: kernel inward, control planes outward, contracts in between.
- Existing open debt remains valid, but cross-stack work should now map back to one shared architectural direction instead of drifting into unrelated local cleanups.
- Dashboard backend, operator, and CLI changes that depend directly on router internals are now recognized as architectural debt by default unless the dependency is intentionally part of a public contract seam.
- Harness docs and local rules must stay aligned with the real execution hotspots, especially in the CLI where orchestration ownership has moved away from `docker_cli.py`.
- This decision does not force an immediate rewrite. It sets the ratchet for future loops: shrink shared sinks, make runtime composition explicit, and replace deep imports with narrow contract or service seams.
