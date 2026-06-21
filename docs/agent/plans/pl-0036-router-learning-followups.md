# Router Learning Follow-Up Closure

## Goal

Implement the remaining follow-up work from
`website/docs/proposals/router-learning-memory-and-adaptations.md`: shared
Router Learning runtime contracts, experience, adaptation composition,
stateful algorithm migration, distributed-states guidance, architecture eval,
AMD validation, and a GitHub PR.

## Scope

- Keep the public Router Learning API clean-break only.
- Keep request-time routing fail-open and free of synchronous external storage
  reads.
- Treat Router Replay as the durable event log.
- Treat states as low-latency runtime memory owned by enabled adaptations.
- Treat experience as materialized historical evidence generated from replay,
  eval, or overrides and read from local snapshots.
- Move learning-style selector states into Router Learning adaptations.
- Validate local behavior before AMD deployment validation.

## Exit Criteria

- A second adaptation can run through the same internal input/output contract
  used by `session_aware`.
- Headers and replay can represent multiple learning adaptations and the final
  composed decision.
- Experience has an internal materialized snapshot contract and fail-open
  diagnostics.
- `elo`, `rl_driven`/Thompson, and `gmtrouter` no longer expose public
  decision-local learning algorithm config; their migrated capabilities live
  under Router Learning adaptations.
- Bandit day0 supports `linucb` by default and `linear_thompson` as a supported
  mode, with weighted `quality`, `cost`, and `latency` goals.
- Distributed-states semantics are documented in deployment guidance.
- Architecture eval reports route correctness, base/final model, adaptation
  method/mode/action/reason/scope, cache/cost delta, unnecessary switch rate,
  bypass behavior, replay coverage, and p50/p95 overhead.
- Local gates pass.
- AMD validation passes against the agentic recipe and captures routing,
  learning, replay, dashboard, and API evidence.
- The final branch is pushed and a GitHub PR is opened.

## Task List

- [x] RLF-001 Rebase the proposal follow-up plan onto latest `origin/main`.
- [x] RLF-002 Add a shared Router Learning runtime contract and deterministic
  adaptation composer.
- [x] RLF-003 Move learning headers and replay diagnostics to method-keyed
  multi-adaptation output.
- [x] RLF-004 Add internal experience snapshot contracts and diagnostics.
- [x] RLF-005 Add global/decision config for `bandit`, `elo`, and
  `personalization` adaptations across Go router and Python CLI schema.
- [x] RLF-006 Migrate bandit day0 (`linucb`, `linear_thompson`) into Router
  Learning states and replay update paths.
- [x] RLF-007 Migrate Elo states into Router Learning states and feedback paths.
- [x] RLF-008 Migrate GMTRouter personalization states into Router Learning.
- [x] RLF-009 Reject old public `algorithm.type` paths for migrated learning
  algorithms with actionable errors.
- [x] RLF-010 Add architecture eval fixtures and reports.
- [x] RLF-011 Update tutorials, proposal status, and deployment/distributed
  states guidance.
- [x] RLF-012 Run local gates and fix failures.
- [x] RLF-013 Run AMD deployment validation.
- [x] RLF-014 Push branch and open PR.

## Next Action

Monitor PR review and follow-up comments. Local focused tests,
`make agent-validate`, `make agent-lint`, `make vllm-sr-test`,
`make test-semantic-router`, and `make agent-ci-gate` passed before AMD
deployment validation. AMD validation ran on PR head `6c162e31` with the
`agentic-saars.yaml` recipe and PR images:

- Router Envoy ready on `:8899`, dashboard ready on `:8700`, and the vLLM ROCm
  backend serving the recipe aliases.
- Conversation-scope routing covered simple/local, privacy bypass, high-care
  legal/health, complex/code, independent new conversations, and tool-loop
  hard locks with `x-vsr-learning-*` headers.
- Temporary session-scope validation covered first-selected session protection
  across conversations and idle-time release to a stronger model.
- Router Replay captured method-keyed learning diagnostics, and vLLM metrics
  exposed prefix-cache counters.
- PR #2258 is open at
  `https://github.com/vllm-project/semantic-router/pull/2258`; checks are
  green/neutral and merge is blocked only by review policy.

## Operating Rules

- Do not reintroduce config rewrite or compatibility migration for old learning
  algorithm paths.
- Do not make request routing depend on synchronous external storage.
- Do not expose broad public `states.enabled` or `experience.enabled` toggles.
- Keep implementation slices independently testable.
- Record intentional architecture gaps in this plan or indexed tech debt rather
  than leaving them only in chat.

## Related Docs

- `website/docs/proposals/router-learning-memory-and-adaptations.md`
- `docs/agent/plans/pl-0035-router-learning-session-aware.md`
- `docs/agent/change-surfaces.md`
- `docs/agent/module-boundaries.md`
