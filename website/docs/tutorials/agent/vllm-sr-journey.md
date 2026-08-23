---
title: vLLM-SR Contributor Journey
description: Deploy, scaffold recipes, validate contracts, evaluate routing, and activate changes with explicit review and rollback.
---

# vLLM-SR Contributor Journey

This tutorial describes the contributor-facing agent skill and helper commands
for issue [#2977](https://github.com/vllm-project/semantic-router/issues/2977).
Generation and evaluation produce **review artifacts**; activation remains a
separate, explicit step.

## Supported journeys

| Journey | Start | Output |
| --- | --- | --- |
| Local deploy | cpu / amd / nvidia-local | Validated config and canonical serve commands |
| K8s deploy | Helm, Operator, or `make e2e-test` | Validated config plus supported deployment pointers |
| Fork built-in MoM | `vllm-sr model fork vllm-sr/mom-v1-blend` | Customized config with MoM lifecycle notes |
| New maintained recipe | `vllm-sr recipe scaffold --name <name>` | Five-file recipe under `config/recipes/<name>/` |
| Calibrate routing | Probes plus live router | Eval reports via the calibration loop |
| Continuous tuning | Replay in observe mode | Offline `recipe-learning` report for human review |

## Safety model

- Do not invent unsupported deployment paths.
- Keep private hostnames, credentials, and internal endpoints out of committed artifacts.
- Do not auto-promote tuning output to active router config.
- Capture rollback paths before live activation.

## Detect environment

```bash
python3 tools/agent/scripts/vllm_sr_journey.py detect-env
make agent-vllm-sr-journey ARGS="detect-env"
```

The helper reads [`tools/agent/repo-manifest.yaml`](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/repo-manifest.yaml)
and maps to canonical build and serve commands for `cpu-local`, `amd-local`,
`nvidia-local`, and `ci-k8s`.

## Scaffold or fork a recipe

Minimal maintained recipe:

```bash
vllm-sr recipe scaffold --name my-recipe
```

Fork an existing maintained recipe:

```bash
vllm-sr recipe scaffold --name my-recipe --from-recipe balance
```

Fork a built-in catalog asset:

```bash
vllm-sr recipe scaffold --name my-recipe --from vllm-sr/mom-v1-blend
```

Multi-profile scaffold:

```bash
vllm-sr recipe scaffold --name my-recipe --multi-profile
```

See the illustrative example at
[`config/recipes/examples/journey-starter/`](https://github.com/vllm-project/semantic-router/tree/main/config/recipes/examples/journey-starter).

## Validate

```bash
vllm-sr validate --config config/recipes/my-recipe/config.yaml
python3 tools/agent/scripts/vllm_sr_journey.py validate \
  --config config/recipes/my-recipe/config.yaml \
  --recipe-dir config/recipes/my-recipe
```

For maintained catalog recipes, also run `make recipe-conformance-static`.

## Evaluate

Static gates:

```bash
python3 tools/agent/scripts/vllm_sr_journey.py evaluate \
  --config config/recipes/my-recipe/config.yaml
```

Live router probes:

```bash
python3 tools/agent/scripts/vllm_sr_journey.py evaluate \
  --config config/recipes/my-recipe/config.yaml \
  --router-url http://127.0.0.1:8080 \
  --probes config/recipes/my-recipe/probes.yaml
```

Semantic routing calibration against a live apiserver uses
[`tools/agent/scripts/router_calibration_loop.py`](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/scripts/router_calibration_loop.py).

## Review before activation

```bash
python3 tools/agent/scripts/vllm_sr_journey.py review \
  --config config/recipes/my-recipe/config.yaml \
  --recipe-dir config/recipes/my-recipe \
  --output-dir .agent-harness/vllm-sr-journey/my-session/
```

Review bundles include digests, validation receipts, evaluation results,
routing summaries, rollback hints, and `activated: false`.

## Activate explicitly

| Target | Command |
| --- | --- |
| Local | `make agent-serve-local ENV=cpu` or `vllm-sr serve --config ...` |
| Live apiserver | `router_calibration_loop.py deploy ...` with version capture |
| K8s | Helm `configOverride` or Operator CR update |
| Dashboard | Mixture-of-Models workspace activation |

## Continuous tuning loop

1. Enable `global.router.learning` in **observe** mode.
2. Collect Router Replay evidence.
3. Run `vllm-sr eval recipe-learning --output-dir ./report`.
4. Review patch suggestions manually.
5. Re-run journey evaluation before switching sensitive decisions to **apply**.

## Agent skill source of truth

The executable skill lives at
[`tools/agent/skills/contributor/vllm-sr-journey/SKILL.md`](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/contributor/vllm-sr-journey/SKILL.md).

Related docs:

- [Configuration workflows](../installation/configuration-workflows.md)
- [MoM model family](../overview/mom-model-family.md)
- [Recipe conformance guide](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/CONFORMANCE.md)
