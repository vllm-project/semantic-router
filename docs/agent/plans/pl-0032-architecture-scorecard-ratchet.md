# Architecture Debt Consolidation

## Goal

- Keep exactly one non-release architecture debt plan.
- Own open TD entries that are valuable but not required for the active release.
- Convert scattered architecture cleanup into a small, reviewable queue with
  explicit retirement criteria.

## Scope

- Non-release technical debt:
  - [TD006](../tech-debt/td-006-structural-rule-exceptions.md)
  - [TD016](../tech-debt/td-016-fleet-sim-shared-ruff-contract-gap.md)
  - [TD017](../tech-debt/td-017-fleet-sim-structure-gate-migration-gap.md)
  - [TD020](../tech-debt/td-020-classification-subsystem-boundary-collapse.md)
  - [TD027](../tech-debt/td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md)
  - [TD042](../tech-debt/td-042-ffi-embedding-structure-debt.md)
  - [TD043](../tech-debt/td-043-semantic-router-go-cyclop-debt.md)
  - [TD045](../tech-debt/td-045-runtime-generation-ownership-gap.md)
  - [TD046](../tech-debt/td-046-control-plane-contract-convergence-gap.md)
- [../architecture-scorecard.md](../architecture-scorecard.md)
- Harness scoring and validation logic that decides what is current.

Out of scope:

- v0.3 Themis release closure; that belongs to
  [PL0033](pl-0033-v0-3-themis-release-closure.md).
- Daily GitHub issue and PR operations; that belongs to maintainer ops.
- Plan archaeology.

## Exit Criteria

- Every non-release open TD has one owner plan: this file.
- The scorecard reports only current release/debt work.
- Each debt item is either retired with source evidence or split into a new
  release-owned item when it becomes release-critical.
- No completed work contributes open task count.

## Task List

- [x] `ADC001` Keep only current execution plans in active tracking.
- [x] `ADC002` Assign owner-plan metadata to the current non-release TD set.
- [ ] `ADC003` Retire or narrow TD006 by reducing remaining structural exceptions
  that no longer need special treatment.
- [ ] `ADC004` Retire or narrow TD016 and TD017 by bringing fleet-sim lint and
  structure expectations into the shared contract.
- [ ] `ADC005` Retire or narrow TD020 by extracting classifier orchestration
  seams that are still doing request-time policy work.
- [ ] `ADC006` Retire or narrow TD027 by splitting fleet-sim optimizer analysis,
  verification, and export ownership.
- [ ] `ADC007` Retire or narrow TD042 and TD043 while preserving diff-scoped
  native binding gates.
- [ ] `ADC008` Introduce runtime-generation ownership, leases, rollback, and
  bounded actor shutdown to retire TD045.
- [ ] `ADC009` Converge config ingress and mutation on one strict canonical
  contract and transactional writer to retire TD046.

## Next Action

- Choose one non-release TD and either retire it with current-source evidence or
  split it into a release-bound issue if it blocks the active milestone.

## Operating Rules

- Keep non-release architecture debt out of release planning unless it directly
  blocks a release track.
- Use current-source evidence to decide whether a TD is still open.

## Related Docs

- [Tech Debt README](../tech-debt/README.md)
- [Architecture Scorecard](../architecture-scorecard.md)
