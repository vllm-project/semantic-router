# Maintained Routing Recipes

## Overview

This directory contains complete routing examples for common deployment goals.
Each recipe is documented as a Model Card: it explains the intended use,
request-facing behavior, backend requirements, safety properties, evaluation
scope, and limitations.

A recipe selects among configured provider models. It does not install or
start those inference backends.

## Model Cards

| Recipe | Best for |
| --- | --- |
| [Accuracy](accuracy/README.md) | Direct answers by default, with bounded workflow or fusion when accuracy benefits from orchestration. |
| [Agent](agent/README.md) | Coding, research, specialist, privacy, and security work across local and frontier lanes. |
| [Balanced](balance/README.md) | General-purpose quality, latency, cost, and answer-recovery trade-offs. |
| [Feedback Recovery](feedback/README.md) | Corrections, repeated dissatisfaction, failed code, and verification requests. |
| [Knowledge](knowledge/README.md) | Evidence-based escalation from a small local model to a stronger model. |
| [Multi-Objective](multi-objective/README.md) | Five request-facing balance, speed, cost, accuracy, and privacy profiles over one shared pool. |
| [Privacy-First](privacy/README.md) | Local containment for sensitive and suspicious requests. |

The [built-in virtual model catalog](built-in/README.md) is a separate
distribution surface. Its [MoM V1 Model
Card](built-in/latest/mom-v1/README.md) describes the virtual models bundled
with `vllm-sr`.

## Use a recipe

Read the Model Card first, start the required provider backends, then validate
and serve the recipe's config:

```bash
vllm-sr validate --config config/recipes/<name>/config.yaml
vllm-sr serve --config config/recipes/<name>/config.yaml
```

Single-profile recipes use the configured `vllm-sr/auto` entrypoint.
Multi-profile recipes expose named virtual model IDs through top-level
`entrypoints`.

## Use a built-in model

Built-in models can be discovered and selected without locating their source
files:

```bash
vllm-sr model list
vllm-sr model show vllm-sr/mom-v1-blend
vllm-sr serve vllm-sr/mom-v1-blend
```

This starts the local routing stack, not the physical model engines. Use
`model fork` when changing provider bindings or routing policy:

```bash
vllm-sr model fork vllm-sr/mom-v1-blend mom-custom.yaml
vllm-sr model validate mom-custom.yaml
vllm-sr serve --config mom-custom.yaml
```

See the [built-in catalog guide](built-in/README.md) for version selection,
multiple virtual models, and customization.

## Custom recipes and Dashboard

Keep credentials out of recipe files. Reference environment variables from the
config and authorize each required name when serving:

```bash
export PROVIDER_API_KEY=...
vllm-sr serve --config path/to/recipe/config.yaml \
  --recipe-env PROVIDER_API_KEY
```

`vllm-sr recipe pack` is available for teams that need to transport a custom
recipe as an archive. Treat the archive as public source: the packer rejects
literal credentials and unsafe package shapes, but it does not turn the recipe
into a built-in model or provision its runtime dependencies.

When a managed recipe is mounted, Dashboard shows its Model Card and probe
catalog. **Run** sends a probe to Playground, **Edit** prepares an editable
request, and **Validate** evaluates routing without generating a model answer.
See [Models and Recipes](../../website/docs/installation/models-and-recipes.md)
for the user workflow.

## For contributors

Every maintained recipe contains:

- `metadata.yaml` for identity and licensing;
- `config.yaml` for runtime configuration;
- `recipe.dsl` for the reviewable routing projection;
- `probes.yaml` for backend-independent routing scenarios; and
- `README.md` for the Model Card.

The conformance tools discover recipe directories automatically and generate
current coverage results. Do not copy probe inventories, pass counts, or CI
receipts into Model Cards.

Start with [Recipe authoring and conformance](CONFORMANCE.md):

```bash
make recipe-conformance-static
```

Syntax-only DSL examples such as `bounded-candidate-iteration.dsl` are not
deployable recipes and do not belong in the Model Card index.
