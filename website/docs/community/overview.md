# Contributing

Contributions are welcome across the Router, CLI, Dashboard, deployment assets,
documentation, and evaluation tools. The repository's
[CONTRIBUTING.md](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md)
is the authoritative workflow; this page is a short path to the checks most
contributors need.

## Before you start

- Search existing issues and pull requests before opening overlapping work.
- Read the nearest `AGENTS.md` before changing a module with local rules.
- Keep a pull request focused on one behavior or documentation outcome.
- Add or update tests when a change affects observable behavior.

## Local workflow

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router

make agent-bootstrap
make agent-report ENV=cpu CHANGED_FILES="path/to/changed-file"
```

`agent-report` identifies the smallest relevant validation commands for the
changed paths. Use `ENV=amd` only for ROCm-specific behavior.

For the default local image workflow:

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never
```

See the [Development Guide](./development) for targeted tests and runtime
commands.

## Before opening a pull request

Run the checks reported for your change. These repository-wide entrypoints are
useful when their scope matches your work:

```bash
make precommit-check
make agent-ci-gate CHANGED_FILES="path/to/changed-file"
make test-and-build-local
```

Documentation-only changes can use:

```bash
make agent-docs-ci-gate AGENT_BASE_REF=origin/main
```

Every commit in a pull request must include a Developer Certificate of Origin
sign-off:

```bash
git commit -s -m "describe the change"
```

In the pull request, explain the problem, the user-visible result, and the
validation you ran. Include screenshots only when they help reviewers evaluate
a visual change.

## Contributor guides

- [Development Guide](./development)
- [Documentation Guide](./documentation)
- [Code Style and Quality](./code-style)
- [Architecture overview](/docs/overview/semantic-router-overview)
