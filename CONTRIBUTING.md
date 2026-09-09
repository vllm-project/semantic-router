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

Project work has exactly one `wg/*` owner. A bounded parent outcome uses an
`[Epic]` title and the automatically synchronized `epic` label; implementation
issues are linked below it as sub-issues. The retired `area/*` and `track/*`
taxonomies are not part of the repository contract and must not be recreated.

Do not begin a non-trivial implementation or open a PR until the tracking issue
is accepted. PRs must link an accepted issue with exactly one Workgroup owner;
the Community check enforces this contract.

## Understand the change surface

[AGENTS.md](AGENTS.md) is the short entrypoint to the repository's development
harness. The human-readable index is
[tools/agent/docs/README.md](tools/agent/docs/README.md).

For a non-trivial change, inspect the repository facts for the paths involved:

```bash
make impact ENV=cpu CHANGED_FILES="path/one path/two"
```

`impact` reports owners, related checks, candidate CI jobs, optional profiles,
and available tools; it does not choose a skill or prescribe a development
loop. Read the nearest `AGENTS.md` for directories you touch. Useful references
are:

- [Change surfaces](tools/agent/docs/change-surfaces.md)
- [Testing strategy](tools/agent/docs/testing-strategy.md)
- [Architecture guardrails](tools/agent/docs/architecture-guardrails.md)

If implementation and intended architecture still differ after your change,
open an owned GitHub issue. Use the compact
[architecture risk index](tools/agent/docs/architecture-risks.md) only when a
repository-local boundary must remain visible before an issue exists.

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

Run the daily changed-file check first:

```bash
make check CHANGED_FILES="path/one path/two"
```

It checks formatting/lint and deterministic dependency/generated contracts
without editing source. Apply formatting explicitly with `make fmt`. Choose
tests that prove the change; CI runs the affected suites in their owning jobs.
Common direct targets include:

| Change | Command |
| --- | --- |
| Harness or workflows | `make harness-check` |
| Go router | `make test-semantic-router` |
| Native bindings | `make test-binding` |
| Python CLI | `make vllm-sr-test` |
| Category, PII, or jailbreak classifier | `make test-category-classifier`, `make test-pii-classifier`, or `make test-jailbreak-classifier` |
| Explicit integration or E2E | `make verify DOMAIN=<domain>` or `make verify PROFILE=<profile>` |

Integration and E2E are explicit because a path classifier cannot infer all
runtime intent. Use the local core baseline for high-risk changes or when
a reviewer asks for it:

```bash
make ci-full
```

This runs containerized quality checks and the local core test/build suite.
It does not reproduce hardware-specific jobs or qualify a release.

A failed gate is part of the work: fix the cause and rerun the smallest
relevant command until it passes.

## Code quality

Install the repository hooks once:

```bash
make precommit-install
```

Run the branch check on demand with:

```bash
make check
```

Follow the language's standard formatter and keep modules focused:

- Go: `gofmt`, meaningful exported API comments, and `make check-go-mod-tidy`.
- Rust: `cargo fmt`, `cargo clippy`, explicit error handling, and public API
  documentation.
- Python: Ruff-compatible formatting, type hints where they improve the
  interface, and tests for behavior changes.

Behavior changes need tests that prove them. Use integration or E2E when
correctness depends on interaction across components; pure refactors do not
require new E2E coverage. Do not add a second source of truth for schemas,
test selection, or public documentation.

## Submit a pull request

1. Link the change to an accepted issue with exactly one `wg/*` owner.
2. Create a focused branch and make one coherent change.
3. Update tests, examples, and public docs for behavior the user can observe.
4. Run the relevant checks and record the commands and outcomes in the
   PR template.
5. Commit with a Developer Certificate of Origin sign-off:

   ```bash
   git commit -s -m "describe the change"
   ```

6. Open a PR using the module prefixes and sections in
   [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).

Mergify places a pull request in the merge queue only after at least two
reviewers with `write`, `maintain`, or `admin` repository permission approve
it and all required checks pass.

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
