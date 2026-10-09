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
- To claim `ready-for-dev` work, comment `/assign` on the issue. The claim
  counts once your name appears under Assignees; comment `/unassign` to
  release it. Assignments on issues that are not yet accepted are removed.
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

`impact` reports owners, minimum checks, candidate CI jobs, optional profiles,
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
VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest \
  vllm-sr serve --image-pull-policy never
```

The build tags local images as `latest`. Set `VLLM_SR_IMAGE` explicitly because
an editable CLI installation with a stable package version defaults to that
release's image tag. The CLI derives the official Dashboard image with the same
tag; `--image-pull-policy never` prevents pulling missing images.

For the AMD local image:

```bash
make vllm-sr-dev VLLM_SR_PLATFORM=amd
VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest \
  vllm-sr serve --image-pull-policy never --platform amd
```

If you customize `DOCKER_TAG`, `DOCKER_REGISTRY`, or the Make image variables,
pass the actual built images to `serve` through `VLLM_SR_IMAGE` and, when needed,
`VLLM_SR_DASHBOARD_IMAGE`. The build's completion message prints a startup
command with the selected images.

Use `vllm-sr logs <service>`, `vllm-sr status`, and `vllm-sr stop` to inspect
and stop the stack.

## Test your change

Run the daily changed-file check first:

```bash
make check CHANGED_FILES="path/one path/two"
```

It runs formatting/lint, deterministic architecture contracts, and the owning
domains' smallest unit or static checks. Common direct targets include:

| Change | Command |
| --- | --- |
| Harness or workflows | `make harness-check` |
| Go router | `make test-semantic-router` |
| Model runtime | `make model-runtime-test` |
| Python CLI | `make vllm-sr-test` |
| Published models (the Vela 1.0 classifiers and the Vela 2.0 decision models) | `make test-models` |
| Explicit integration or E2E | `make verify DOMAIN=<domain>` or `make verify PROFILE=<profile>` |

Integration and E2E are explicit because a path classifier cannot infer all
runtime intent. Use the complete local baseline for high-risk changes or when
a reviewer asks for it:

```bash
make ci-full
```

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

Python formatting checks report trailing whitespace, file-ending, and Black
formatting issues without rewriting source. To apply the native Python formatters
deliberately, run:

```bash
.venv-agent/bin/pre-commit run python-format --hook-stage manual --files path/to/file.py
```

Pre-commit exits nonzero when a formatter changes files. Review the diff, then
rerun `make check`.

Follow the language's standard formatter and keep modules focused:

- Go: `gofmt`, meaningful exported API comments, and `make check-go-mod-tidy`.
- Python: Black formatting, type hints where they improve the
  interface, and tests for behavior changes.

Behavior-visible config, routing, CLI, Docker, startup, or API changes require
matching E2E coverage unless they are pure refactors. Do not add a second source
of truth for schemas, test selection, or public documentation.

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

Keep commits reviewable and avoid unrelated cleanup. A PR should explain why
the change is needed, which modules it affects, and how its user-visible
behavior was verified.

Mergify places a pull request in the merge queue only after at least two
reviewers with `write`, `maintain`, or `admin` repository permission approve
it and all required checks pass.

### Troubleshoot the merge queue

Check the **Mergify Merge Queue** check or Mergify's **Merge Queue Status**
comment on the PR. The **Reason** and **Hint** explain whether the PR is
waiting or was dequeued because its branch could not be updated.

- **Waiting in the queue:** the queue processes PRs serially, so a long wait
  can be normal. See [.mergify.yml](.mergify.yml) for the current settings.
  Avoid pushing while queued: a push restarts PR Gate, which the queue requires.
- **Workflow permission refusal:** until
  [#3902](https://github.com/vllm-project/semantic-router/issues/3902) is resolved,
  Mergify cannot update a branch when the update includes workflow changes
  requiring its pending `workflows` permission. An organization owner must
  accept that permission in the Mergify dashboard. Contributors can use the
  manual branch update below; requeueing without updating repeats the failure.
- **Fork cannot be queued:** on a personal-account fork containing workflows,
  GitHub offers **Allow edits and access to secrets by maintainers**. Enabling
  it lets maintainers update the branch, including workflows, which can expose
  fork secrets and allow access to other branches. This is optional; if you
  leave it off, update the branch yourself after a dequeue. See
  [GitHub's explanation of fork permissions][fork-permissions].

To recover from a branch-update failure, merge upstream `main` into your PR
branch and push it, or use GitHub's **Update branch** button when available.

If that merge includes workflow changes, an HTTPS push using an OAuth token
or personal access token (classic) without the `workflow` scope can also be
refused. Use **Update branch**, push over SSH, or use GitHub's **Sync fork**
to update your fork's `main` before retrying the push. GitHub permits workflow
files without that scope when the same paths and contents already exist on
another branch in the fork. See [GitHub's scope documentation][oauth-scopes].

Once PR Gate passes on the new head and the required approvals still hold,
the rule requeues the PR automatically. The `@mergifyio queue` command and
**Requeue** checkbox require write permission by default; contributors do not
need them for this recovery.

[fork-permissions]: https://docs.github.com/en/pull-requests/how-tos/work-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork
[oauth-scopes]: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps

## Repository map

| Path | Responsibility |
| --- | --- |
| `src/semantic-router/` | Go router, config, routing, APIs, and Envoy ExtProc service |
| `src/vllm-sr/` | Python CLI and local stack orchestration |
| `src/model-runtime/` | Model runtime that serves every classifier, embedding, and decision model |
| `config/` | Canonical reference, fragments, runtime examples, and Recipes |
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
