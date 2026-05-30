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

- API, config, deployment, and control-plane contracts.
- Stable versions, upgrades, and production operations.
- Runtime state, durability, recovery, and telemetry.
- Performance and native backend parity on real hardware.
- Research that feeds product: session affinity and model-selection quality.
- Product hardening across dashboard, eval, RAG/memory, protocol surfaces, and
  ClawOS-related control surfaces that are already part of v0.3.
- Observability and E2E only where they validate the release tracks above.

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
- Security and RBAC hardening. Security bugs can still be triaged as normal
  bugfixes, but new security/RBAC closure is not a v0.3 release track.
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

- [ ] `THEMIS004` Freeze the public config and control-plane contract for v0.3.

  Do:

  - decide which config, DSL, API, operator, CLI, and dashboard projection
    shapes are supported in v0.3
  - remove or reject unsupported shapes with clear validation errors
  - align samples, docs, generated config, CRD/webhook behavior, and dashboard
    deploy paths to the same contract

  Done when: TD015, TD028, and TD039 are retired, narrowed, or accepted as
  explicit release risks.

- [ ] `THEMIS005` Make release-critical state restart-safe.

  Do:

  - list the runtime and dashboard state that must survive restart for v0.3
  - choose the durable owner for each kept state surface, or mark it explicitly
    ephemeral
  - verify deployed config, dashboard control-plane config, runtime reload, and
    local dev integration behavior after restart

  Done when: TD031, TD034, and TD037 are retired, narrowed, or accepted as
  explicit release risks.

- [ ] `THEMIS006` Ship only the dashboard and eval hardening needed by v0.3.

  Do:

  - narrow the dashboard config/editor flows needed for release
  - make training or evaluation artifacts that feed product flows use one
    documented contract
  - add focused dashboard or eval validation for the kept workflows
  - defer exploratory dashboard, eval, and product ideas that do not block v0.3

  Done when: TD030 and TD032 are retired, narrowed, or accepted as explicit
  release risks.

- [ ] `THEMIS007` Publish and smoke the native backend support matrix.

  Do:

  - state which Candle and ONNX backend features are supported in v0.3
  - make unsupported native paths fail early with actionable errors
  - verify native package, image, startup, reset, and upgrade or rollback claims
    against the support matrix

  Done when: TD033 is retired, narrowed, or accepted as an explicit release
  risk.

- [ ] `THEMIS008` Decide the v0.3 session-aware model-switching scope.

  Do:

  - choose one release answer: ship a small session-aware routing slice, defer
    it, or close stale attempts and re-open a smaller issue
  - do not treat old stale PRs as the source of truth unless they are rebased
    and reduced to the chosen scope
  - document the release decision and any accepted user-visible limitation

  Done when: the milestone and PR queue reflect the chosen session-routing
  decision.

- [ ] `THEMIS009` Produce the final release readiness result.

  Do:

  - run the release validation gates for the kept tracks
  - update scorecard evidence and maintainer release readiness output
  - assemble release notes from merged PRs, closed issues, and accepted risks

  Done when: every kept v0.3 task is closed, narrowed, or explicitly accepted
  as a release risk.

## Next Action

- Decide which of `THEMIS004` through `THEMIS008` are kept for v0.3, which are
  cut, and which are accepted as release risks. Start with `THEMIS004`, because
  the contract baseline affects the other tracks.

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
