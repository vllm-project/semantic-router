---
name: vllm-sr-journey
category: support
description: Guides deployment, recipe generation, contract validation, evaluation, and reviewed tuning for vLLM-SR. Use when a contributor needs to detect a supported environment, scaffold or fork a recipe, validate artifacts, run eval gates, or prepare an activation-ready change without auto-promoting it.
---

# vLLM-SR Contributor Journey

## Trigger

- Use when a contributor needs to deploy vLLM-SR locally or on supported K8s paths
- Use when user intent should become a valid v0.3 config or maintained five-file recipe
- Use when generated artifacts must be validated and evaluated before activation
- Use when tuning should stay in observe mode until a human reviews replay or probe evidence
- Use when the task needs a review bundle with provenance, diff, rollback hints, and an explicit not-activated flag

## Required Surfaces

- contributor_interface
- local_smoke

## Conditional Surfaces

- harness_exec
- docs_examples
- dsl_crd
- k8s_platform
- deployment_profile_stack

## Stop Conditions

- The target environment is unsupported or the workflow would invent a deployment path
- Validation fails and the root cause is not classified or recorded
- A live deploy would run without a captured rollback version or git revert path
- Private hostnames, credentials, or internal endpoints would be committed to public artifacts
- Tuning output would be auto-promoted to production or active router config without review

## Workflow

1. Detect the target environment and choose a supported path only.
   - Read [`tools/agent/repo-manifest.yaml`](../../../../../tools/agent/repo-manifest.yaml) `supported_envs` and run `python3 tools/agent/scripts/vllm_sr_journey.py detect-env`.
   - Local paths stay on the canonical image flow: `make vllm-sr-dev`, then `vllm-sr serve --image-pull-policy never`.
   - K8s paths defer to Helm, Operator, or `make e2e-test`; do not invent alternate serve or deploy commands.
2. Generate or update artifacts from intent, not from unsupported shortcuts.
   - Quick local config: start from [`src/vllm-sr/cli/templates/config.template.yaml`](../../../../../src/vllm-sr/cli/templates/config.template.yaml) or fork a built-in model with `vllm-sr model fork`.
   - Maintained recipe: run `vllm-sr recipe scaffold --name <name> [--from <catalog-model> | --from-recipe <recipe>]`.
   - Compose signals, decisions, algorithms, and plugins from [`config/fragments/`](../../../../../config/fragments/) when extending policy.
3. Explain the routing design before asking for activation.
   - Summarize the model pool, signals, decisions, algorithms, plugins, backend assumptions, and credential/env requirements.
   - For MoM or built-in virtual models, respect the lifecycle in the MoM docs; routing policy does not install checkpoints.
4. Validate against repository contracts before evaluation or activation.
   - Run `vllm-sr validate --config <path>`.
   - For maintained recipes, run `python3 tools/agent/scripts/vllm_sr_journey.py validate --config <path> --recipe-dir <dir>`.
   - Keep private endpoints redacted; use placeholders such as `host.docker.internal:8000` and document `--recipe-env` bindings in the Model Card Requirements section.
5. Run or guide evaluation before activation.
   - Static-only: recipe conformance static checks and manifest inventory review.
   - Live router: `python3 tools/agent/scripts/vllm_sr_journey.py evaluate --config <path> --router-url <url> [--probes <manifest>]`.
   - Semantic routing tuning with a live apiserver: delegate to [`routing-calibration-loop`](../../maintainer/routing-calibration/SKILL.md).
   - Continuous model-choice tuning: keep `global.router.learning` in `observe`, collect replay, then run `vllm-sr eval recipe-learning`; never auto-merge the report.
6. Produce a review bundle instead of activating by default.
   - Run `python3 tools/agent/scripts/vllm_sr_journey.py review --config <path> [--recipe-dir <dir>] [--output-dir .agent-harness/vllm-sr-journey/<session>/]`.
   - Include digests, validation receipts, eval results, routing summary, rollback hints, and `activated: false`.
7. Activate only after explicit human approval.
   - Local: `make agent-serve-local ENV=cpu|amd` or `vllm-sr serve --config <path>`.
   - Live apiserver: `router_calibration_loop.py deploy` with version capture.
   - K8s: Helm `configOverride` or Operator CR update from the reviewed canonical YAML.
   - Dashboard: Mixture-of-Models workspace activation after separate provider and routing checks.

## Gotchas

- Generation is not acceptance. A valid scaffold or fork still needs review, eval, and an explicit activation step.
- `vllm-sr serve` starts the routing stack only; provider backends in `providers.models` must already be reachable.
- Recipe probe success does not prove backend generation quality; verify provider endpoints separately.
- Do not commit `.agent-harness/vllm-sr-journey/` review bundles or private environment details into the repo.
- Examples under `config/recipes/examples/` are illustrative; maintained catalog recipes live as direct children of `config/recipes/`.

## Must Read

- [tools/agent/docs/environments.md](../../../../../tools/agent/docs/environments.md)
- [config/recipes/CONFORMANCE.md](../../../../../config/recipes/CONFORMANCE.md)
- [website/docs/installation/configuration-workflows.md](../../../../../website/docs/installation/configuration-workflows.md)
- [website/docs/overview/mom-model-family.md](../../../../../website/docs/overview/mom-model-family.md)

## Standard Commands

- `python3 tools/agent/scripts/vllm_sr_journey.py detect-env`
- `python3 tools/agent/scripts/vllm_sr_journey.py validate --config <path> [--recipe-dir <dir>]`
- `python3 tools/agent/scripts/vllm_sr_journey.py evaluate --config <path> [--router-url <url>] [--probes <manifest>]`
- `python3 tools/agent/scripts/vllm_sr_journey.py review --config <path> [--recipe-dir <dir>] [--output-dir <dir>]`
- `vllm-sr recipe scaffold --name <name> [--from <catalog-model> | --from-recipe <recipe>]`
- `vllm-sr validate --config <path>`
- `make agent-serve-local ENV=cpu|amd`
- `make agent-vllm-sr-journey`

## Acceptance

- The journey selects only supported environments and canonical deploy commands
- Generated or forked artifacts pass `vllm-sr validate` and, for maintained recipes, static recipe conformance checks
- Evaluation runs before activation and failures are classified instead of patched blindly
- Review bundles record provenance, receipts, rollback hints, and an explicit not-activated state
- No private environment details appear in committed artifacts
