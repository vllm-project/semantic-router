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

Every maintained recipe includes an `omni` decision backed by an explicit
image-content signal. Bind that decision to a configured visual-language Model
when publishing the Entrypoint; text-only decisions remain unchanged.

The [built-in Recipe distribution](built-in/README.md) is a separate,
Recipe-only surface. Its [MoM V1 Model
Card](built-in/latest/mom-v1/README.md) describes the routing profiles bundled
with `vllm-sr`.

## Use a recipe

Read the Model Card first and start the required provider backends. Then start
Semantic Router once:

```bash
vllm-sr serve
```

In the Dashboard, connect the physical Models, choose or import the Recipe,
assign Models to its decisions, and publish an Entrypoint. An independent
control plane can perform the same lifecycle through the Router Management API.
The CLI does not select or materialize a Recipe at launch time.

## Use a built-in Recipe

Start Semantic Router, then open **Recipes** in the Dashboard:

```bash
vllm-sr serve
```

Connect provider endpoints in **Models**, choose a built-in Recipe, create a
**Mixture of Models**, and assign configured Models to its decisions before
publishing an Entrypoint. The Dashboard uses the Router Management API, so an
independent control plane can perform the same lifecycle. See the
[built-in Recipe guide](built-in/README.md) for the packaged assets.

## Custom recipes and Dashboard

Keep credentials and provider endpoints out of recipe files. Manage provider
connections separately, then create or import the Recipe through the Dashboard
or Router Management API.

`vllm-sr recipe pack` is available for teams that need to transport a custom
recipe as an archive. Treat the archive as public source: the packer rejects
literal credentials and unsafe package shapes, but it does not install the
Recipe as a built-in distribution resource or provision runtime dependencies.

When a managed Recipe is mounted, Dashboard shows its Model Card and probes.
**Run** sends a probe to Playground, **Edit** prepares an editable
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
