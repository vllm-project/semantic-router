# Design Elegance Ratchet Execution Plan

## Goal

- Turn the latest maintainability review into executable, resumable cleanup work.
- Ratchet the repository toward deeper modules, narrower interfaces, and more honest acceptance tests.
- Close TD019, TD020, and TD021 once the code and harness reflect those standards.

## Scope

- `docs/agent/architecture-guardrails.md`
- `docs/agent/testing-strategy.md`
- `docs/agent/feature-complete-checklist.md`
- `docs/agent/module-boundaries.md`
- `docs/agent/local-rules.md`
- `docs/agent/repo-map.md`
- `e2e/testcases/**`
- `src/semantic-router/pkg/classification/**`
- `src/semantic-router/pkg/services/classification.go`
- Milvus-backed runtime stores under `src/semantic-router/pkg/{memory,cache,vectorstore,routerreplay,responsestore}/`

## Exit Criteria

- TD019, TD020, and TD021 are all closed with concrete code and validation evidence.
- Baseline E2E behavior contracts use meaningful thresholds or are explicitly reclassified as benchmarks/reporting-only coverage.
- Classification runtime seams are split enough that discovery/bootstrap, per-family inference, and service assembly are not coupled through the same hotspot orchestrators.
- Milvus lifecycle logic is factored behind a reusable seam that is adopted across the duplicated runtime stores.

## Task List

- [x] `S001` Record the new maintainability debt in canonical TD entries and create this execution plan.
- [x] `S002` Codify the review outcomes in shared guardrails and local `AGENTS.md` files so future edits inherit the new standards.
- [x] `S003` Audit the `ai-gateway` and shared router testcase set into acceptance contracts versus report-only or benchmark cases, then define explicit thresholds for the acceptance set.
- [x] `S004` Refactor shared E2E testcase helpers so metric collection, detailed reporting, and pass/fail evaluation are separate concerns.
- [ ] `S005` Extract classification discovery/bootstrap, per-family adapters, and service assembly seams so `classifier.go` and `services/classification.go` stop growing as multi-role orchestrators.
- [ ] `S006` Introduce a shared Milvus lifecycle seam and migrate the duplicated runtime stores toward it without collapsing their domain-specific query semantics.
- [ ] `S007` Run the required harness validation and close TD019, TD020, and TD021 when the executable code catches up to the desired design.

## Current Loop

- 2026-03-18: shared acceptance thresholds landed for the baseline `ai-gateway` router contract via `e2e/pkg/testcases`, and the changed-file harness gate passed.
- 2026-03-18: cold-start `ai-gateway` local E2E timed out during profile install because the profile had to anonymously download required Hugging Face models without `HF_TOKEN`; treat that run as an environment constraint, not evidence against the shared acceptance seam.
- Next loop target: start `S005` by extracting classifier discovery/bootstrap and family-boundary seams out of the current hotspot orchestrators.

## Decision Log

- Treat classic code-quality guidance as a ratchet on repository-specific seams, not as generic style prose.
- Prefer local `AGENTS.md` files for directory-specific rules and shared harness docs for cross-cutting design principles.
- Do not close a debt item when only the documentation improves; closure requires executable code or validation behavior to converge.

## Follow-up Debt / ADR Links

- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../tech-debt/td-019-e2e-behavior-contract-thresholds.md](../tech-debt/td-019-e2e-behavior-contract-thresholds.md)
- [../tech-debt/td-020-classification-subsystem-boundary-collapse.md](../tech-debt/td-020-classification-subsystem-boundary-collapse.md)
- [../tech-debt/td-021-milvus-adapter-duplication-across-runtime-stores.md](../tech-debt/td-021-milvus-adapter-duplication-across-runtime-stores.md)
- [../adr/README.md](../adr/README.md) (no dedicated ADR yet)
