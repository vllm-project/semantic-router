# TD032: Training and Evaluation Artifact Contracts Still Drift Across Dashboard, Runtime, and Scripts

## Status

Open

## Scope

`src/training/**`, dashboard backend training or evaluation runners, and runtime selection or evaluation seams that consume training-generated artifacts

## Summary

The repository now exposes training, evaluation, and model-research flows through both scripts and dashboard APIs, but the production-facing contract for those artifacts is still informal. `src/training/model_eval/result_to_config.py` directly emits a runtime-facing canonical config scaffold with hard-coded model-catalog, prompt-guard, classifier, and tools defaults. Dashboard backend subsystems launch training and evaluation through subprocess calls into `src/training/**`, persist ad hoc YAML or JSON outputs, and infer progress from command output. Runtime selection packages and CLI help text then refer back to those training outputs and external verifier servers such as `automix_verifier.py` and `router_r1_server.py`, but the repo lacks one typed artifact manifest describing which outputs are experimental, which are production-consumable, how they are versioned, and how dashboard/runtime code should resolve them. Path resolution has already drifted at least once: dashboard router fallback logic still guesses `src/training/ml_model_selection` while the actual package lives under `src/training/model_selection/ml_model_selection`.

## Evidence

- [src/training/model_eval/result_to_config.py](../../../src/training/model_eval/result_to_config.py)
- [src/training/model_eval/test_result_to_config.py](../../../src/training/model_eval/test_result_to_config.py)
- [src/semantic-router/pkg/modelselection/selector.go](../../../src/semantic-router/pkg/modelselection/selector.go)
- [src/semantic-router/pkg/selection/automix.go](../../../src/semantic-router/pkg/selection/automix.go)
- [src/semantic-router/pkg/selection/rl_driven.go](../../../src/semantic-router/pkg/selection/rl_driven.go)
- [dashboard/backend/mlpipeline/runner.go](../../../dashboard/backend/mlpipeline/runner.go)
- [dashboard/backend/modelresearch/loop.go](../../../dashboard/backend/modelresearch/loop.go)
- [dashboard/backend/evaluation/runner.go](../../../dashboard/backend/evaluation/runner.go)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [src/vllm-sr/cli/commands/runtime_support.py](../../../src/vllm-sr/cli/commands/runtime_support.py)
- [website/docs/training/model-performance-eval.md](../../../website/docs/training/model-performance-eval.md)

## Why It Matters

- Adding a new training-backed algorithm, benchmark, or provider requires coordinated edits across training scripts, runtime loaders, dashboard launchers, docs, and sometimes CLI messaging because there is no shared artifact contract.
- Hard-coded defaults in training-to-config scaffolds can silently diverge from canonical runtime defaults, model-catalog semantics, or deployment-specific constraints.
- Dashboard control-plane code currently treats long-running training and evaluation flows as subprocess conventions, which weakens timeout, retry, provenance, and resumability behavior as those workloads grow.
- Contributors cannot easily tell which outputs under `src/training/**` are experimental notebooks/scripts, which are dashboard-managed workflows, and which are intended to feed steady-state production config.

## Desired End State

- Training and evaluation outputs that can feed runtime or dashboard workflows are described by a typed manifest or artifact contract with ownership, schema, and lifecycle rules.
- Dashboard training, evaluation, and model-research backends resolve scripts, output directories, and metadata through one shared adapter layer instead of per-feature subprocess conventions.
- Runtime selection and evaluation paths consume named artifact contracts rather than inferring semantics from loose file paths, ad hoc JSON, or external service URLs.
- Docs clearly distinguish experimental workflows from production promotion paths for training-backed artifacts.

## Exit Criteria

- The repo has one documented and validated contract for training-produced artifacts that feed runtime, dashboard, or CLI workflows.
- Dashboard backends no longer hard-code divergent training-path guesses or per-feature subprocess conventions for the same training stack.
- Training-generated config fragments or model metadata can round-trip into runtime-facing validation without relying on ad hoc default injection.
- New training-backed routing features can be added by extending the shared artifact contract rather than rediscovering script paths and output schemas in multiple modules.
