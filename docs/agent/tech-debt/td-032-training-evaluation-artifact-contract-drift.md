# TD032: Training and Evaluation Artifact Contracts Still Drift Across Dashboard, Runtime, and Scripts

## Status

Closed

## Scope

`src/training/model_eval/**`, `dashboard/backend/{router,evaluation,mlpipeline}/**`, `src/semantic-router/pkg/{trainingartifacts,selection}/**`, and training docs that define production-consumable artifacts

## Summary

The repository now exposes one shared training artifact contract instead of relying on scattered path guesses and hard-coded scaffold defaults. `src/semantic-router/pkg/trainingartifacts/contract.json` and `CurrentContract()` define the production-consumable training scripts, benchmark/train output names, external verifier service metadata, and runtime-facing scaffold defaults used by dashboard backends and `result_to_config.py`. Dashboard evaluation and ML pipeline runners now resolve project roots, script paths, output directories, and model artifact names through that contract, while runtime selection surfaces use the same contract for named `automix` and `router_r1` service metadata. The previous path drift to `src/training/ml_model_selection` and the duplicated runtime-default injection in `result_to_config.py` are retired.

## Evidence

- [src/training/model_eval/result_to_config.py](../../../src/training/model_eval/result_to_config.py)
- [src/training/model_eval/test_result_to_config.py](../../../src/training/model_eval/test_result_to_config.py)
- [src/semantic-router/pkg/trainingartifacts/contract.json](../../../src/semantic-router/pkg/trainingartifacts/contract.json)
- [src/semantic-router/pkg/trainingartifacts/contract.go](../../../src/semantic-router/pkg/trainingartifacts/contract.go)
- [src/semantic-router/pkg/trainingartifacts/contract_test.go](../../../src/semantic-router/pkg/trainingartifacts/contract_test.go)
- [src/semantic-router/pkg/selection/automix.go](../../../src/semantic-router/pkg/selection/automix.go)
- [src/semantic-router/pkg/selection/rl_driven.go](../../../src/semantic-router/pkg/selection/rl_driven.go)
- [dashboard/backend/mlpipeline/runner.go](../../../dashboard/backend/mlpipeline/runner.go)
- [dashboard/backend/mlpipeline/runner_contract_test.go](../../../dashboard/backend/mlpipeline/runner_contract_test.go)
- [dashboard/backend/evaluation/runner.go](../../../dashboard/backend/evaluation/runner.go)
- [dashboard/backend/evaluation/runner_test.go](../../../dashboard/backend/evaluation/runner_test.go)
- [dashboard/backend/router/core_routes.go](../../../dashboard/backend/router/core_routes.go)
- [dashboard/backend/router/core_routes_test.go](../../../dashboard/backend/router/core_routes_test.go)
- [website/docs/training/model-performance-eval.md](../../../website/docs/training/model-performance-eval.md)
- [website/docs/training/ml-model-selection.md](../../../website/docs/training/ml-model-selection.md)

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

- Satisfied on 2026-04-06: the repo has one documented and validated contract for training-produced artifacts that feed runtime, dashboard, or CLI-adjacent workflows.
- Satisfied on 2026-04-06: dashboard backends no longer hard-code divergent training-path guesses or duplicate benchmark/train artifact naming for the same training stack.
- Satisfied on 2026-04-06: training-generated config scaffolds now read runtime-facing defaults from the shared artifact contract instead of ad hoc default injection.
- Satisfied on 2026-04-06: new training-backed routing features can extend the shared artifact contract rather than rediscovering script paths and output schemas across modules.

## Resolution

- Added `src/semantic-router/pkg/trainingartifacts/{contract.json,contract.go}` as the shared manifest and typed Go adapter for model-eval scripts, ML pipeline artifacts, runtime-facing defaults, and named external verifier services.
- `dashboard/backend/router/core_routes.go`, `dashboard/backend/evaluation/runner.go`, and `dashboard/backend/mlpipeline/**` now resolve project roots, script locations, output file names, train directories, cache directories, and generated values filenames through that shared contract.
- `src/training/model_eval/result_to_config.py` now reads scaffold defaults from the shared contract, and its tests assert the generated config remains in sync with the contract-owned defaults.
- `src/semantic-router/pkg/selection/{automix.go,rl_driven.go}` now surface contract-owned metadata for the external AutoMix verifier and Router-R1 services instead of relying on loose script-path comments.
- Training docs now distinguish contract-owned production artifacts from experimental `src/training/**` content.

## Validation

- `pytest /Users/bitliu/vs/src/training/model_eval/test_result_to_config.py`
- `go test ./pkg/trainingartifacts ./pkg/selection`
  Run in `/Users/bitliu/vs/src/semantic-router`
- `go test ./router ./evaluation ./mlpipeline`
  Run in `/Users/bitliu/vs/dashboard/backend`
