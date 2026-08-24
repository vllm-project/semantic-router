# Agent Routing Recipe Model Card

## Overview

Agent Routing sends coding, research, specialist, security, and privacy work to
different model lanes. Sensitive or suspicious requests stay local, simple
requests use a local fast path, and work that needs broader capability can move
to specialist or frontier backends.

The recipe records replay data for its routed decisions. Router Learning can
use that data when enabled, but learning is not required to run the recipe.

## Model details

| Lane | Reference models |
| --- | --- |
| Local containment and fast path | `qwen/qwen3.6-rocm` |
| Fast specialist work | `google/gemini-2.5-flash-lite` |
| Complex code and research | `google/gemini-3.1-pro`, `openai/gpt5.4` |
| Legal and health | `anthropic/claude-opus-4.6` |

Clients use `vllm-sr/auto`. The names above are logical provider aliases and
can be rebound to compatible backends.

## Intended use

Use this recipe for:

- coding assistants and tool-using agents;
- technical research and STEM questions;
- business, legal, or health specialist routing;
- private-context requests that must remain on a local lane; and
- deployments collecting approved replay data for later policy improvement.

It is not a complete agent runtime: the router selects a model and applies
route policy, while the client remains responsible for executing tools and
managing application state.

## Routing behavior

| Request pattern | Behavior |
| --- | --- |
| Jailbreak or private-context signal | Stay on the local model and bypass learning adaptation. |
| Simple math or general request | Use the local fast path. |
| Coding, research, business, legal, or health request | Use the corresponding specialist lane. |
| Complex request without a specialist domain | Escalate to a frontier lane. |
| Unmatched request | Fall back to the simple local lane. |

Domain and general-purpose routes include exclusion guards so one request does
not accidentally match several peer lanes.

## Requirements

- Reachable OpenAI-compatible endpoints for the five configured aliases.
- Embedding, category, PII, and complexity classifiers.
- A Postgres replay store when replay collection is enabled.
- `POSTGRES_PASSWORD` supplied from the environment rather than committed to
  the recipe.

The repository ships no Postgres password. `vllm-sr serve` generates a distinct
credential per stack for the Postgres service it manages, so this binding
matters when the recipe points at a Postgres you run yourself. Use a separately
managed credential and database for production.

## Data handling and safety

Replay is enabled and stored in Postgres for 30 days by the checked-in config.
It captures request and response bodies for replay-enabled routes: up to 2 KiB
per body on containment, privacy, and simple-math routes, and up to 4 KiB on
the other specialist routes. This includes requests routed through the local
security and privacy lanes.

Treat the replay database as sensitive application data. Restrict access,
encrypt it as required by the deployment, and shorten retention or disable body
capture when prompts and responses must not be stored.

## Quick start

```bash
export POSTGRES_PASSWORD="$(openssl rand -base64 24)"  # your own value
vllm-sr validate --config config/recipes/agent/config.yaml
vllm-sr serve --config config/recipes/agent/config.yaml \
  --recipe-env POSTGRES_PASSWORD
```

## Evaluation

The probes exercise security containment, privacy policy, specialist domains,
complexity bands, fallbacks, and replay-enabled routes. They verify routing and
plugin selection without calling the configured inference backends. See
[`probes.yaml`](probes.yaml) and the [conformance guide](../CONFORMANCE.md).

## Limitations

- Classifier quality directly affects routing quality.
- Replay storage contains captured request and response data under the
  checked-in policy; review it before using the recipe with sensitive traffic.
- The checked-in provider aliases share a development endpoint and must be
  rebound for a real heterogeneous deployment.
- The DSL represents the routing graph, while YAML-only adaptation settings in
  `config.yaml` remain part of the complete runtime policy.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
