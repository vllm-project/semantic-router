# Claude Code Instructions

This file configures Claude Code for the vLLM Semantic Router repository.

## Repository Overview

vLLM Semantic Router is an intelligent request router for LLM inference that sits as an Envoy ExtProc filter. It classifies incoming requests and routes them to appropriate model backends based on semantic signals (reasoning complexity, PII, jailbreak detection, etc.).

### Key Subsystems

- `src/semantic-router/` — Go router (Envoy ExtProc server, config, routing logic)
- `src/vllm-sr/` — Python CLI for local dev workflow
- `candle-binding/`, `ml-binding/`, `nlp-binding/` — Rust inference bindings
- `dashboard/` — Web UI (React frontend + Go backend)
- `deploy/operator/` — Kubernetes operator and CRDs
- `e2e/` — End-to-end test framework (kind/Kubernetes)
- `tools/agent/` — Agent harness manifests and scripts
- `tools/make/` — Canonical Make targets

## Development Workflow

### Local Setup

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never
```

For AMD ROCm:

```bash
make vllm-sr-dev VLLM_SR_PLATFORM=amd
vllm-sr serve --image-pull-policy never --platform amd
```

### Validation Commands (use smallest gate first)

```bash
make agent-validate          # harness-only changes
make agent-lint CHANGED_FILES="path/one,path/two"
make agent-ci-gate CHANGED_FILES="path/one,path/two"
make agent-pr-gate           # full local PR baseline
make test-and-build-local    # reproduces CI Test And Build job
make agent-feature-gate ENV=cpu CHANGED_FILES="path/one,path/two"
```

### Running Tests

```bash
make test-binding            # Rust bindings
make test-semantic-router    # Go router
make test-category-classifier
make test-pii-classifier
make test-jailbreak-classifier
make e2e-test                # full E2E (requires kind cluster)
```

### Linting

```bash
make go-lint                 # Go lint
make go-lint-fix             # Go lint with auto-fix
make check-go-mod-tidy       # verify Go modules
pre-commit run --all-files   # all pre-commit hooks
markdownlint -c tools/linter/markdown/markdownlint.yaml "**/*.md"  # Markdown lint
```

## Commit Trajectory and PR Scoping

### Commit Structure

Structure commits as a reviewable trajectory — each commit should represent one logical step that builds on the previous:

1. **Separate concerns into distinct commits**: refactoring in one commit, new logic in the next, tests in another.
2. **Each commit must compile and pass lint**: never leave the tree broken mid-sequence.
3. **Commit messages describe the "why"**: e.g., `extract helper to reduce hotspot responsibility` not `move code`.
4. **Sign off all commits**: `git commit -s -m "message"` (DCO required).

### PR Scoping

- **Narrow the blast radius**: touch only the files and subsystems necessary for the change. Do not mix unrelated cleanups or drive-by fixes into a feature PR.
- **One subsystem per PR when possible**: a Router change should not also refactor the Dashboard unless they are tightly coupled.
- **PR titles use module-aligned prefixes**: `[Router]`, `[Dashboard]`, `[Docs]`, `[CI/Build]`, `[Operator]`, etc.
- **Include a Test Plan section**: describe what was validated and how.
- Behavior-visible routing, startup, config, Docker, CLI, or API changes require E2E test updates unless pure refactor.

### Example Commit Trajectory

```
commit 1: [Router] extract PII signal helper from processor hotspot
commit 2: [Router] add configurable PII threshold to signal helper
commit 3: [Router] add unit tests for PII threshold edge cases
commit 4: [Router] update E2E test for new PII threshold behavior
```

This makes review incremental — reviewers can follow the reasoning step by step.

## Architecture Rules

- **Layer model**: `signal` → `decision` → `algorithm` → `plugin` → `global`
- Keep modules narrow: one main responsibility per file.
- Legacy hotspots are debt, not precedent. Do not grow their responsibility.
- Interfaces belong only at true seams.
- Read the nearest local `AGENTS.md` before editing hotspot directories.

### Hotspot Directories (check local AGENTS.md first)

- `src/semantic-router/pkg/config/`
- `src/semantic-router/pkg/extproc/`
- `src/vllm-sr/cli/`
- `deploy/operator/api/v1alpha1/`
- `deploy/operator/controllers/`
- `dashboard/frontend/src/`
- `dashboard/backend/handlers/`

## Agent Harness

This repo has a full agent harness. For complex tasks:

1. Read `AGENTS.md` for the entrypoint.
2. Read `docs/agent/README.md` for the full system of record.
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to get routed context.
4. Validate with the canonical gates before marking work complete.

The executable rule layer lives in:

- `tools/agent/repo-manifest.yaml`
- `tools/agent/task-matrix.yaml`
- `tools/agent/skill-registry.yaml`
- `tools/agent/structure-rules.yaml`
- `tools/make/agent.mk`
