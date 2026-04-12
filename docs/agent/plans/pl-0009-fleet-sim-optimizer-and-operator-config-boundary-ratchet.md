# Fleet Sim Optimizer and Operator Config Boundary Ratchet Execution Plan

## Goal

- Turn the latest design audit into executable, resumable cleanup work for the fleet-sim optimizer surface and the Kubernetes operator config contract.
- Reduce responsibility collapse in `fleet_sim/optimizer/base.py`, the fleet-sim public export layers, and the operator CRD plus canonical-config translation path.
- Retire TD027 and TD028 only after code, local rules, and validation evidence converge on narrower seams.

## Scope

- `src/fleet-sim/fleet_sim/optimizer/**`
- `src/fleet-sim/fleet_sim/__init__.py`
- `deploy/operator/api/v1alpha1/**`
- `deploy/operator/controllers/canonical_config_builder.go`
- shared harness docs and local `AGENTS.md` files for these hotspot trees

## Exit Criteria

- TD027 and TD028 are both closed with concrete code and validation evidence.
- Fleet-sim optimizer changes no longer collapse analytical sizing, DES verification, flexibility or TPW analysis, and public export policy into one hotspot seam.
- Operator config changes have clearer boundaries between CRD schema declaration, webhook validation, canonical config translation, and sample-fixture maintenance.
- Shared and local AGENT rules stay aligned with the active hotspot boundaries so future changes inherit the narrower design target.

## Task List

- [x] `S001` Record the fleet-sim optimizer and operator config contract debts in canonical TD entries and create this execution plan.
- [x] `S002` Tighten shared and local AGENT rules so the optimizer and operator hotspot trees are explicit before code extraction starts.
- [ ] `S003` Split fleet-sim optimizer ownership so analytical sizing, DES calibration, power or flexibility analysis, and report helpers stop growing in one `base.py`.
- [ ] `S004` Narrow fleet-sim export ownership so `optimizer/__init__.py` and `fleet_sim/__init__.py` stop mirroring an overly broad optimizer surface.
- [ ] `S005` Split operator config ownership so CRD schema families, webhook validation helpers, controller-side canonical translation, and sample-fixture expectations stop collapsing into the current hotspots.
- [ ] `S006` Run the required harness and subsystem validation for the remaining work and close TD027 and TD028 when the narrowed seams are proven.

## Current Loop

- 2026-03-19: the latest design audit identified two additional subsystem-specific boundary gaps not yet captured by TD016/TD017 or TD006: fleet-sim optimizer/public-surface collapse and operator config-contract collapse.
- 2026-03-19: TD027 and TD028 were added so those active hotspots are no longer only implicit structural debt.
- 2026-03-19: shared and local AGENT rules were updated so future edits discover the fleet-sim optimizer and operator API/controller hotspot trees before widening them further.
- 2026-04-03: ADR0006 now provides the shared cross-stack dependency direction for the remaining operator-side work in this plan; operator follow-up should converge on contract-owned seams rather than deeper router-runtime dependencies.
- Next loop target: start `S003` by separating optimizer analytical sizing, DES verification, and report-oriented helpers into narrower sibling modules with stable exports.

## Decision Log

- Prefer repo-local AGENT rules and execution plans over a new repo-local skill when the guidance is tightly coupled to one subtree's ownership and hotspots.
- Treat TD006 and TD017 as structural ratchets, not as substitutes for subsystem-specific debt when the repo needs a clearer architecture target for active modules.
- Keep fleet-sim optimizer refactoring and operator config-contract refactoring in the same plan because both emerged from the same latest design audit and both need durable governance plus follow-up code extraction.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../tech-debt/td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md](../tech-debt/td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md)
- [../tech-debt/td-028-operator-config-contract-boundary-collapse.md](../tech-debt/td-028-operator-config-contract-boundary-collapse.md)
- [../adr/adr-0006-platform-kernel-and-contract-first-control-planes.md](../adr/adr-0006-platform-kernel-and-contract-first-control-planes.md)
