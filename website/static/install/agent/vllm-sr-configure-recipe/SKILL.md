---
name: vllm-sr-configure-recipe
description: Turn a workload and reachable answer-model backends into a reviewable vLLM Semantic Router recipe and entrypoint configuration proposal. Use when someone asks which maintained recipe fits or wants a canonical config candidate without activating it.
---

# Propose a Router recipe

Produce a candidate canonical v0.5 document and an evidence-backed diff. A
proposal is neither persisted nor active. Use the installed CLI and the
[configuration procedure](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md)
as the authority; do not invent a second generator or silently apply a change.

## 1. Establish the baseline

Inspect the current file, running config, and packaged recipes before asking
the user to repeat information already available. If a Router is running, get
its active document and config identity; local `config.yaml` may differ from
`.vllm-sr/runtime-config.yaml` or a Dashboard edit.

```bash
vllm-sr status
vllm-sr config get --format yaml --endpoint "$ROUTER_ORIGIN" > active.yaml
vllm-sr recipe builtin list
vllm-sr config schema --section recipes
```

Use `config get` only when the management endpoint exists. For an unstarted
stack, read its complete config file and validate against the CLI's bundled
schema. Record file digest or active `config_version`, schema/release channel,
and the intended public entrypoint. Preserve the baseline bytes for review.
If the document is legacy or incomplete, report the migration or missing
fields; do not make a v0.5 proposal by guessing.

Gather only missing facts: representative request shape (messages, tools,
modalities, context and output size), quality/cost/latency objective and guard
floors, available answer-model endpoints and their verified capabilities,
prices and limits, security/data-placement constraints, and target platform.
Check endpoint reachability from where the Router will run. Credentials stay in
environment variables or referenced files, never in the proposal.

## 2. Select a supported path

Inspect a candidate with `vllm-sr recipe builtin export BUNDLE --output-dir
DIR` and its maintained [recipe card](https://github.com/vllm-project/semantic-router/tree/main/config/recipes).
Choose a single recipe when one policy covers the workload; use separate
entrypoints when objectives or policy boundaries differ. A concrete provider
model name bypasses recipe routing. See
[Entrypoints and recipes](https://vllm-sr.ai/docs/tutorials/global/entrypoints-and-recipes).

Map each recipe decision to actual provider model aliases. Confirm its
`minimum_candidates`, input/output modality, context, tool and reasoning
requirements; do not substitute a backend that lacks a required capability.
Check which model-backed signals and System One selectors it asks with the
[decision-model skill](https://vllm-sr.ai/install/agent/vllm-sr-decision-model/SKILL.md).
If the installed CLI supports the desired bounded change, use
`vllm-sr config propose --help` and its maintained intent. Otherwise edit a
copy of the canonical document. State unsupported combinations and uncertainty
instead of asserting an unmeasured model ranking.

## 3. Build and check the candidate

Copy the complete baseline to a separate `candidate.yaml`. Change only the
requested `recipes[]` and `entrypoints[]` fields unless a missing provider or
deployment is explicitly part of the request. Preserve listeners, providers,
evaluation, global settings, existing recipes and unrelated entrypoints. Show
a redacted unified diff, the source recipe/fragment version, and each observed
backend fact behind the recommendation.

```bash
vllm-sr config validate --config candidate.yaml --endpoint "$ROUTER_ORIGIN"
vllm-sr config plan --config candidate.yaml --mode replace --endpoint "$ROUTER_ORIGIN"
```

With no live Router, run `config validate --config candidate.yaml` and report
that only offline checks completed; `config plan` requires a running management
endpoint. Treat a CLI message that only structural checks ran as limited
evidence. For a live stack, require plan `valid: true`, inspect `changed`,
restart requirements, deployment loads and any warnings. `changed: false`
may mean the candidate is already active; it is not proof of a new proposal.
Do not run `config apply`, `serve --replace-active-config`, or edit the active
runtime file during this workflow.

## 4. Present the handoff

Give the baseline identity, workload and backend assumptions, chosen recipe
and why it fits, candidate path, exact diff, validation and plan receipts,
unresolved capability or endpoint checks, and the state of the live stack.
The human reviewer decides whether to apply or activate it through the
[configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#change-a-running-stack).
After activation, [preview then probe](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md)
the actual entrypoint; a valid config alone says nothing about route quality.

## Controlled two-backend example

The [base fixture](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-base.yaml)
has `fast` and `strong` answer-model aliases and one default recipe. The
[candidate fixture](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-candidate.yaml)
keeps the listener and both providers intact, adds a `code` recipe using
`strong`, and publishes `example/code` while keeping `vllm-sr/auto` as the
default. The endpoints are placeholders: verify actual backends before live
planning or serving. This example demonstrates the diff and canonical parse,
not a claim that `strong` is empirically better for code or that a Router is
running.
