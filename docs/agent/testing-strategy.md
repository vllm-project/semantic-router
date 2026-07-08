# Testing Strategy

This document defines the harness-side validation ladder for repository changes.

## Validation Ladder

- `make agent-validate`
  - use for harness-only changes
  - validates manifests, docs inventory, rule layering, and link portability
- `make agent-scorecard`
  - shows the current harness inventory and whether validation is passing
- `make agent-lint CHANGED_FILES="..."`
  - runs pre-commit, language lint, and structure checks for changed files
  - Go changed-file lint reuses stricter module configs when the repository defines them; `dashboard/backend` uses the same `golangci-lint` config as `make dashboard-lint`
- `make agent-ci-lint CHANGED_FILES="..."`
  - reproduces the CI changed-file lint path locally
  - runs `make codespell-tracked` and `make agent-fast-gate` with the same agent bootstrap toolchain used by CI
- `make precommit-branch-gate`
  - reproduces the local prelint gate that the repo installs as a `pre-push` hook
  - runs `make agent-ci-lint` followed by `make precommit-check`
- `make agent-ci-gate CHANGED_FILES="..."`
  - runs `agent-report`, `agent-fast-gate`, and rule-driven fast tests
- `make agent-pr-gate`
  - reproduces the baseline PR requirements locally
  - runs the CI-style pre-commit path plus the local reproduction of `Test And Build`
- `make test-and-build-local`
  - reproduces the CI `Test And Build` job locally
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`
  - runs the CI gate, feature tests, local smoke when required, and a final report
- `make agent-e2e-affected CHANGED_FILES="..."`
  - explicit manual local E2E path for affected profiles when debugging or additional local confidence is needed

`CHANGED_FILES` accepts comma-separated, whitespace-separated, or newline-separated
paths. For long changed-file lists or paths that need exact shell preservation, write
the paths one per line and pass `AGENT_CHANGED_FILES_PATH=<file>` instead.

## Selection Rules

- Harness-only prose or manifest changes start with `make agent-validate`.
- Code changes start with `make agent-report` to resolve primary skill, impacted surfaces, and validation commands.
- Use the smallest gate that matches the change.
- Use `make agent-ci-lint CHANGED_FILES="..."` when you want the same changed-file lint path that the pre-commit workflow runs in CI.
- Use `make precommit-branch-gate` when you want the same local prelint gate the installed `pre-push` hook runs before a push or PR update.
- Use `make agent-pr-gate` before opening or updating a PR when you want the repo-native local baseline for the same CI jobs contributors most often miss.
- Use `ENV=amd` when platform behavior, AMD defaults, or ROCm image selection are affected.

## Loop Handling

- `make agent-report` resolves the task loop mode from `tools/agent/task-matrix.yaml`.
- `lightweight` tasks keep iterating until the applicable gates for the current change pass or a real external blocker is recorded.
- `completion` tasks keep iterating until the active subtask passes its applicable gates; when the work spans multiple ordered loops or sessions, move the task state into an execution plan.
- A failing lint, test, smoke, integration, or E2E run is not a handoff point. Fix the issue or narrow the failing surface, then rerun the smallest relevant gate until the current completion boundary is satisfied.

## Environment Expectations

- `cpu-local`
  - default local path for feature work
- `amd-local`
  - required for AMD-specific behavior and real-model deployment validation
- `ci-k8s`
  - merge-gate coverage and profile matrix validation

See [environments.md](environments.md) for the concrete commands.

## Behavior and Coverage Rules

- Behavior-visible routing, startup, config, Docker, CLI, or API changes require updated or new E2E coverage unless the change is a pure refactor.
- Documentation-only changes should not trigger local smoke or heavy E2E unless the task matrix escalates them.
- Core, common, startup-chain, Docker, or agent-execution changes may expand CI profile coverage beyond the locally affected set.
- Local E2E remains available, but it is an explicit manual path instead of part of the default `agent-feature-gate`.
- Workflow-driven integration suites are part of the canonical validation story when they are listed in `tools/agent/e2e-profile-map.yaml`.
- The current workflow-driven suites are:
  - `vllm-sr-cli-integration` via `make vllm-sr-test-integration`
  - `memory-integration` via `make memory-test-integration`
- Manual-only Go profiles are valid durable suites, but they must be named in `manual_profile_rules` instead of existing as undocumented runner-only paths.

## Model-Gated Multimodal Tests

Tests that need the real `multi-modal-embed-small` model gate on the
`MULTIMODAL_MODEL_PATH` environment variable and skip when it is unset.
The `Test And Build` workflow (`test-and-build` job) exports
`MULTIMODAL_MODEL_PATH` pointing at `models/mom-embedding-multimodal`, which
`make download-models` fetches earlier in the same `make test` invocation, so
the following suites execute on every `test-and-build` run (PRs matching the
job's path filters - core, make, ci, helm, e2e, docker - non-draft only,
pushes to `main`, and the nightly schedule; docs-only changes do not fire the
job):

- `TestEmbeddingClassifier_Integration*` in
  `src/semantic-router/pkg/classification/` (hermetic synthetic-PNG
  image-encode path through the real candle-binding FFI), via
  `test-semantic-router`.
- The hermetic candle-binding multimodal Go tests
  (`TestMultiModalEmbeddingInit`, `TestMultiModalEncodeText`,
  `TestMultiModalInputValidation`), via `test-binding-minimal`.

Still manual, split into two targets so exit status stays meaningful:

- `make test-binding-multimodal` - the pass/fail receipt for PRs touching the
  multimodal FFI: all multimodal binding Go tests plus the router integration
  tests. This includes the image-encode Go tests
  (`TestMultiModalEncodeImageFromBytes/Base64/URL`,
  `TestMultiModalCrossModalRetrieval`), which download fixture images from
  Wikimedia at test time and therefore stay out of the CI lane to avoid
  external-network flakiness. The same image-encode FFI path is covered
  hermetically in CI by the `pkg/classification` integration tests above.
  A non-zero exit from this target means the change broke something.
- `make test-binding-multimodal-rust-baseline` - the exploratory lane for the
  `#[ignore = "requires model files"]` Rust unit tests in
  `candle-binding/src/model_architectures/embedding/multimodal_embedding.rs`.
  No automated lane runs model-dependent `cargo test --lib`; several of these
  also fetch Wikimedia fixtures. This target has a KNOWN-RED baseline and is
  expected to exit non-zero. Known baseline as of 2026-07: 25 of 32 pass;
  the audio-path tests fail because `WhisperEncoder::load` fails and the
  loader's `.ok()` fallback silently disables the audio encoder,
  `test_text_batch_encoding` hits a broadcast shape mismatch in the batched
  text-encode path, and `test_text_semantic_similarity` asserts a semantic
  ranking the model does not reliably produce (the Go equivalent only logs
  it). These are pre-existing defects surfaced by actually running the suite;
  compare your run against this baseline rather than trusting exit status.

## Acceptance Versus Reporting

- Acceptance tests must encode a meaningful failure condition for the behavior they claim to protect.
- `0%`-only accuracy checks, "at least one request succeeded", or purely report-only success-rate summaries do not satisfy the repository's acceptance bar for routing, classification, plugin, fallback, or API behavior.
- If a testcase is primarily benchmarking, soak, or observability-oriented, keep it explicitly named as reporting coverage unless it also declares an executable threshold.
- When a threshold is probabilistic, document a conservative floor and keep the rationale close to the testcase or shared helper constant.
- Prefer shared helpers that collect metrics and separate helpers that evaluate acceptance, so future testcases do not silently inherit report-only semantics.

## Source of Truth

- Gate selection and commands: [../../tools/agent/task-matrix.yaml](../../tools/agent/task-matrix.yaml)
- Environment resolution: [../../tools/agent/repo-manifest.yaml](../../tools/agent/repo-manifest.yaml)
- E2E profile mapping: [../../tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
- E2E taxonomy and suite selection: [../../tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
- Executable entrypoints: [../../tools/make/agent.mk](../../tools/make/agent.mk)
- Done criteria: [feature-complete-checklist.md](feature-complete-checklist.md)
- Local testcase rules: [../../e2e/testcases/AGENTS.md](../../e2e/testcases/AGENTS.md)
