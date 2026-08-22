# Contributing to vLLM Semantic Router

Thank you for contributing. This guide covers the repository workflow; the
public installation guide is at <https://vllm-sr.ai/docs/installation/>.

## Before you start

Install:

- Docker or Podman;
- Make;
- Git; and
- Python 3.10 or newer for the CLI, tests, or training tools you plan to use.

Clone the repository:

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
```

There is no repository-wide Python `requirements.txt`. Install dependencies
from the subsystem you are changing, for example:

```bash
pip install -r src/vllm-sr/requirements.txt
pip install -r e2e/testing/requirements.txt
```

## Get work accepted before implementation

New feature and bug reports enter `needs-acceptance`. Opening an issue,
receiving reactions, or assigning yourself does not make the work part of the
roadmap.

The issue lifecycle is:

```text
needs-acceptance -> accepted -> ready-for-dev -> in-progress -> closed
```

- A Maintainer applies `accepted` after confirming the goal, scope, roadmap
  fit, and exactly one owning `wg/*` label.
- `accepted` work may remain in the backlog until it is sufficiently specified
  and has review capacity.
- `ready-for-dev` marks accepted, unassigned work that contributors may claim.
- Assignment moves accepted work to `in-progress`.
- `help wanted` and `good first issue` are curated subsets of
  `ready-for-dev`; they are not intake or acceptance labels.
- A release milestone is a time-bound commitment and is applied only after
  acceptance.

Do not begin a non-trivial implementation or open a PR until the tracking issue
is accepted. PRs must link an accepted issue with exactly one Workgroup owner;
the Community check enforces this contract.

## Understand the change surface

[AGENTS.md](AGENTS.md) is the short entrypoint to the repository's development
harness. The human-readable index is
[tools/agent/docs/README.md](tools/agent/docs/README.md).

Before a non-trivial change, ask the harness which subsystem rules and tests
apply:

```bash
make agent-report ENV=cpu CHANGED_FILES="path/one,path/two"
```

Then read the nearest `AGENTS.md` for any hotspot you touch. Useful design and
test references are:

- [Module boundaries](tools/agent/docs/module-boundaries.md)
- [Change surfaces](tools/agent/docs/change-surfaces.md)
- [Testing strategy](tools/agent/docs/testing-strategy.md)
- [Feature-complete checklist](tools/agent/docs/feature-complete-checklist.md)

If implementation and intended architecture still differ after your change,
record the durable gap under
[tools/agent/docs/tech-debt/](tools/agent/docs/tech-debt/README.md) rather than
leaving it only in a PR discussion.

## Run the local stack

Use the repository's local image workflow:

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never
```

For the AMD local image:

```bash
make vllm-sr-dev VLLM_SR_PLATFORM=amd
vllm-sr serve --image-pull-policy never --platform amd
```

Use `vllm-sr logs <service>`, `vllm-sr status`, and `vllm-sr stop` to inspect
and stop the stack.

## Test your change

Start with the tests returned by `agent-report`. Common targets are:

| Change | Command |
| --- | --- |
| Harness or repository structure | `make agent-validate` |
| Go router | `make test-semantic-router` |
| Native bindings | `make test-binding` |
| Python CLI | `make vllm-sr-test` |
| Category, PII, or jailbreak classifier | `make test-category-classifier`, `make test-pii-classifier`, or `make test-jailbreak-classifier` |
| Affected local E2E profiles | `make agent-e2e-affected CHANGED_FILES="..."` |

Use the repository gates before submitting:

```bash
make agent-ci-lint CHANGED_FILES="path/one,path/two"
make agent-ci-gate CHANGED_FILES="path/one,path/two"
```

`make agent-pr-gate` reproduces the baseline PR checks. Use
`make agent-feature-gate ENV=cpu CHANGED_FILES="..."` when the harness reports
feature tests or local smoke as required. Platform-specific changes use the
matching environment, such as `ENV=amd`.

For a docs-only change, the focused gate is:

```bash
make agent-docs-ci-gate AGENT_BASE_REF=origin/main
```

A failed gate is part of the work: fix the cause and rerun the smallest
relevant command until it passes.

## Code quality

Install the repository hooks once:

```bash
make precommit-install
```

Run the branch preflight on demand with:

```bash
make precommit-branch-gate
```

Follow the language's standard formatter and keep modules focused:

- Go: `gofmt`, meaningful exported API comments, and `make check-go-mod-tidy`.
- Rust: `cargo fmt`, `cargo clippy`, explicit error handling, and public API
  documentation.
- Python: Ruff-compatible formatting, type hints where they improve the
  interface, and tests for behavior changes.

Behavior-visible config, routing, CLI, Docker, startup, or API changes require
matching E2E coverage unless they are pure refactors. Do not add a second source
of truth for schemas, test selection, or public documentation.

## Submit a pull request

1. Link the change to an accepted issue with exactly one `wg/*` owner.
2. Create a focused branch and make one coherent change.
3. Update tests, examples, and public docs for behavior the user can observe.
4. Run the harness-selected tests and record the commands and outcomes in the
   PR template.
5. Commit with a Developer Certificate of Origin sign-off:

   ```bash
   git commit -s -m "describe the change"
   ```

6. Open a PR using the module prefixes and sections in
   [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).

Keep commits reviewable and avoid unrelated cleanup. A PR should explain why
the change is needed, which modules it affects, and how its user-visible
behavior was verified.

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/semantic-router/` | Go router, config, routing, APIs, and Envoy ExtProc service |
| `src/vllm-sr/` | Python CLI and local stack orchestration |
| `config/` | Canonical reference, fragments, runtime examples, and Recipes |
| `candle-binding/`, `ml-binding/`, `nlp-binding/`, `onnx-binding/` | Native inference bindings |
| `dashboard/` | Web console frontend and management backend |
| `deploy/` | Helm, operator, Kubernetes, OpenShift, and local deployment assets |
| `e2e/` | End-to-end framework and profiles |
| `src/training/` | Training and evaluation utilities |
| `tools/` | Build, release, CI, smoke, and agent tooling |
| `website/` | Public documentation |

## Get help

Use the [documentation](https://vllm-sr.ai/), GitHub Discussions, or an issue
with a minimal reproduction. Security reports follow
[SECURITY.md](SECURITY.md), not the public issue tracker.

By contributing, you agree that your work is licensed under Apache 2.0.
