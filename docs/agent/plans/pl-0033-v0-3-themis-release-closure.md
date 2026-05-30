# v0.3 Themis Release Closure

## Goal

- Track all current v0.3 Themis release work in one maintainer-owned plan.
- Convert the Themis roadmap into release tracks, issue groups, and closure
  criteria.
- Keep unrelated architecture debt out of the release unless it blocks a
  Themis track.

Themis is the stability-at-scale release. The local roadmap source is
`website/blog/2026-03-12-v0-3-themis-roadmap.md`.

## Scope

Release tracks:

- API, config, and deployment contracts.
- Stable versions, upgrades, and production operations.
- Runtime state, durability, recovery, and telemetry.
- Performance and native backend parity on real hardware.
- Research that feeds product: DSL generation, feedback loops, session
  affinity, and model-selection quality.
- Product hardening across dashboard, eval, RAG/memory, protocol surfaces, and
  ClawOS-related control surfaces.
- Security, RBAC, observability, and E2E closure.

Owned TD entries:

- [TD015](../tech-debt/td-015-weakly-typed-config-and-dsl-contracts.md)
- [TD028](../tech-debt/td-028-operator-config-contract-boundary-collapse.md)
- [TD030](../tech-debt/td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
- [TD031](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [TD032](../tech-debt/td-032-training-evaluation-artifact-contract-drift.md)
- [TD033](../tech-debt/td-033-native-binding-runtime-parity-and-lifecycle-gap.md)
- [TD034](../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- [TD037](../tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md)
- [TD039](../tech-debt/td-039-control-plane-contract-ownership-collapse.md)

Out of scope:

- Non-release hotspot and fleet-sim debt owned by
  [PL0032](pl-0032-architecture-scorecard-ratchet.md).
- Daily maintainer sync, report generation, and seed issue creation; these
  belong to [Maintainer Ops](../maintainer-ops.md).
- Plan archaeology.
- Daily issue and PR status snapshots, which are generated under
  `.agent-harness/maintainer/`.

## Exit Criteria

- GitHub milestone issues map to these release tracks.
- Every release-bound issue has a track, priority, owner, and validation
  expectation.
- Every release-owned TD is retired, narrowed, or explicitly accepted as a
  known release risk.
- Release notes can be assembled from the plan, milestone state, and merged PRs.
- `make agent-scorecard` reports only current release and debt tasks.

## Task List

- [ ] `THEMIS004` Close API/config/deployment contract gaps covering TD015,
  TD028, and TD039.
- [ ] `THEMIS005` Close runtime state and control-plane durability gaps
  covering TD031, TD034, and TD037.
- [ ] `THEMIS006` Close dashboard, eval, and product-hardening gaps covering
  TD030 and TD032.
- [ ] `THEMIS007` Close native/runtime parity gaps covering TD033.
- [ ] `THEMIS008` Decide the final status of session-aware model-switch offline
  updates and balance-recipe calibration for this release.
- [ ] `THEMIS009` Run release validation, update scorecard evidence, and produce
  the final release readiness summary.

## Next Action

- Choose the remaining v0.3 release tracks that are worth doing, then narrow
  each kept track into GitHub issue ownership and exit criteria.

## Operating Rules

- One release means one release plan. Feature work that belongs to v0.3 is
  tracked here, even when implementation happens through GitHub issues.
- Maintainer operations do not belong in this task list unless they become
  release-blocking product work.
- Technical debt enters the release only when it blocks a release track or is
  explicitly accepted as a release risk.
- Daily GitHub state is generated under `.agent-harness/maintainer/`, not stored
  as canonical repo documentation.

## Related Docs

- [Tech Debt README](../tech-debt/README.md)
- [Maintainer Ops](../maintainer-ops.md)
- [Themis roadmap blog](../../../website/blog/2026-03-12-v0-3-themis-roadmap.md)
