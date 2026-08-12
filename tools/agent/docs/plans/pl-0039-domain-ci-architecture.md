# PL-0039 Domain CI Architecture

## Goal

Implement issue
[#2787](https://github.com/vllm-project/semantic-router/issues/2787) as one
reviewable CI architecture change: classify PR changes once, dispatch affected
domain workflows in parallel, preserve stable merge-gate contexts, and unify
nightly and release publication behind explicit lifecycle orchestrators.

## Scope

- Organize validation and publication workflows by domain while keeping
  lifecycle entry workflows few and explicit.
- Add one pull-request dispatcher with shared change classification, affected
  domain calls, and an aggregate gate.
- Preserve branch-protection contexts `Run pre-commit hooks check file lint`
  and `test-and-build`, plus Mergify operator contexts `Lint`, `Unit Tests`,
  `Verify Manifests`, and `Validate OLM Bundle`.
- Consolidate Docker PR, nightly, and release behavior around explicit
  production, developer-companion, and test-fixture inventories with
  affected-image selection for PRs.
- Make stable PyPI publication tag-driven and gate every release publisher on
  one version-contract validation.
- Keep full E2E/operator/image inventories available on nightly, release,
  manual, or explicitly forced runs while retaining affected PR smoke coverage.
- Consolidate compatible maintenance schedules without widening write
  permissions.
- Delete `.github/workflows/skill-review.yml`.
- Update contributor and harness documentation and executable workflow
  validation together.
- Do not change runtime product behavior.

## Exit Criteria

- One PR dispatcher performs change classification exactly once and fans out
  affected reusable domain workflows in parallel.
- Stable required and Mergify check contexts are always reported with success,
  failure, or deliberate skip compatibility for every PR shape.
- PR Docker validation uses an affected-image matrix with downstream build
  receipt subtraction. Nightly retains every production, companion, and test
  image; stable releases contain production deliverables only.
- Exactly one test-gated nightly Docker publication path remains.
- Stable vllm-sr PyPI publishing is tag-driven; manual/nightly development
  publication is explicit.
- Every Docker, Helm, PyPI, crate, and operator release publisher depends on
  successful version-contract validation in `release.yml`.
- AST security scanning has one fork-safe PR implementation and no untrusted
  code executes with a write token under `pull_request_target`.
- Full Kubernetes/operator fanout remains available outside the normal PR path;
  affected PRs retain meaningful smoke coverage.
- Validation permissions default to `contents: read`; write permissions exist
  only on jobs that need them.
- Workflow YAML parses, local reusable-workflow references and contracts
  resolve, job dependency/condition logic is valid, and repo harness gates pass.
- Hosted-CI-only validation gaps are explicitly recorded.

## Task List

- [x] CI-01: Read repository/harness guidance, issue requirements, current
  branch protection, Mergify policy, contributor docs, and ownership policy.
- [x] CI-02: Audit every current workflow and referenced Make/script entrypoint;
  map lifecycle triggers, permissions, check names, and duplicate work.
- [x] CI-03: Add shared PR classification and the domain reusable workflow
  contracts.
- [x] CI-04: Add the PR dispatcher, stable aggregate gate, and compatibility
  check jobs.
- [x] CI-05: Split read-only Docker affected-image validation from
  write-capable main/nightly/release publishing while preserving complete
  inventory parity.
- [x] CI-06: Make release validation the prerequisite for every publisher and
  correct PyPI trigger policy.
- [x] CI-07: Right-size E2E/operator PR smoke and preserve full/manual/nightly
  coverage.
- [x] CI-08: Consolidate safe community/maintenance automation, enforce least
  privilege, replace ownership notification where practical, and quarantine
  hidden moderation implementation.
- [x] CI-09: Update harness classification, local workflow contract tests, and
  contributor/maintainer CI documentation.
- [x] CI-10: Run workflow-focused static checks, then `agent-validate`,
  `agent-lint`, and `agent-ci-gate`; fix and rerun to green.
- [x] CI-11: Replace hidden workflow-local classification with a fixture-tested
  executable contract, remove duplicate binding coverage, and separate
  production release images from nightly companions and fixtures.
- [ ] CI-12: Verify hosted PR check names, fork permissions, affected matrices,
  and nightly/release publisher wiring on the transition PR.

## Next Action

Have the parent review the classifier and lifecycle refinement, then rerun the
transition PR to complete CI-12 without changing branch protection until all
compatibility contexts have reported successfully.

## Hosted-CI Validation Remaining

- Confirm the read-only image validator dispatches without a permission-ceiling
  error. The first hosted attempt exposed and led to fixing the former combined
  validator/publisher contract.
- Confirm the dynamic compatibility jobs report the exact branch-protection
  contexts and `PR Gate` on docs-only, core, dashboard, operator, draft,
  Mergify queue, and fork PRs.
- Confirm `classify_pr_changes.py` receives the expected PR diff range and emits
  the fixture-tested affected job, image, and E2E matrices in GitHub's event
  context.
- Execute affected Docker builds plus the operator/Kubernetes smoke paths on
  hosted runners; local validation did not build large images or create Kind
  clusters.
- Exercise nightly and release workflows with non-publishing test refs or
  reviewed manual validation before the first production tag.
- Content moderation remains deliberately quarantined under
  [TD045](../tech-debt/td-045-reviewed-content-moderation.md); no replacement
  detector was invented.

## Operating Rules

- Keep reusable workflows domain-scoped; do not create one giant workflow.
- Use only static local `uses` references supported by GitHub Actions.
- Preserve stable required-check contexts throughout the migration.
- Keep fork PR validation read-only and do not expose registry credentials.
- Keep publication permissions on publisher jobs, never on broad dispatchers.
- Do not invent moderation behavior when the secret implementation is
  unavailable. Quarantine the unsafe boundary and record any durable remainder
  as indexed technical debt owned by this plan.
- Keep this plan's task status and next action current after every major loop.
- Do not commit, push, create a PR, or mutate GitHub from this worktree.

## Related Docs

- [Agent Harness](../README.md)
- [Testing Strategy](../testing-strategy.md)
- [Maintainer Ops](../maintainer-ops.md)
- [Execution Plans](README.md)
- [Issue #2787](https://github.com/vllm-project/semantic-router/issues/2787)
