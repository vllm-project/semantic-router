# PL-0040 Maintained Recipe Conformance CI

## Goal

Complete [issue #2832](https://github.com/vllm-project/semantic-router/issues/2832):
make every maintained recipe an automatically discovered, executable
conformance contract and integrate the result into the existing PR/Main domain
dispatchers.

## Scope

- Version and strictly validate maintained `probes.yaml` manifests.
- Discover the four-file recipe catalog without a hardcoded recipe list.
- Enforce decision, entrypoint, algorithm, plugin, signal, and alias references.
- Add a compact, Go-native hermetic decision contract matrix.
- Run every base probe against a live CPU router with exact EvalTrace enabled.
- Publish deterministic inventory, coverage, reports, and failure logs.
- Add the reusable Recipe Conformance domain to PR/Main classification and
  aggregate gates.

## Non-goals

- Do not redesign the domain CI dispatcher.
- Do not replace gateway/transport E2E tests.
- Do not require real upstream generation, GPU execution, framing expansion,
  or latency baselines on ordinary pull requests.
- Do not rewrite unrelated integration runners.

## Exit Criteria

- All current recipe directories are discovered with no name allowlist.
- Adding a standard recipe directory automatically enters static and live CI
  planning without workflow edits.
- Every base probe is evaluated through `/api/v1/eval?trace=true`.
- Acceptance and alias mismatches return non-zero.
- Cross-recipe trace leakage fails conformance.
- The hermetic decision matrix and recipe coverage report are deterministic.
- The Recipe Conformance domain participates in `PR Gate` and main validation.
- Canonical harness, workflow, lint, unit, static, and live-CPU gates pass.
- The implementation is merged through a signed-off pull request.

## Task List

- [x] RC-01: Create a clean worktree and branch from current `origin/main`.
- [x] RC-02: Version probe manifests and add strict parsing/schema contracts.
- [x] RC-03: Replace hardcoded recipe names with directory discovery.
- [x] RC-04: Fix calibration exit status, alias matching, and trace validation.
- [x] RC-05: Add inventory, coverage, and shard planning.
- [x] RC-06: Add and validate the Go-native decision contract matrix.
- [x] RC-07: Add local Make targets and live CPU recipe orchestration.
- [x] RC-08: Add reusable workflow and PR/Main domain dispatch.
- [ ] RC-09: Run the complete local validation ladder and fix failures.
- [ ] RC-10: Commit with DCO, create the pull request, and drive CI to green.

## Next Action

Run the complete focused Python/Go/workflow validation, then exercise the live
CPU stack before committing.

## Operating Rules

- Keep the existing domain dispatcher as the CI control plane.
- Keep probe inventory deterministic and materialized before sharding.
- Do not retry assertion failures.
- Keep exact model inference scores and shared-runner latency out of golden
  contracts.
- Preserve raw reports and logs while keeping summaries content-safe.
- Keep this plan current until the PR is merged or the issue is explicitly
  handed off.

## Related Docs

- [Testing strategy](../testing-strategy.md)
- [Maintained recipes](../../../../config/recipes/README.md)
- [Domain CI architecture](pl-0039-domain-ci-architecture.md)
- [Issue #2333](https://github.com/vllm-project/semantic-router/issues/2333)
- [Issue #2379](https://github.com/vllm-project/semantic-router/issues/2379)
