# Journey Starter Recipe Model Card

## Overview

Journey Starter is an illustrative routing recipe for the vLLM-SR contributor
journey. It demonstrates the five-file maintained recipe contract without
joining the CI live-conformance matrix under `config/recipes/`.

## Model details

| Role | Placeholder |
| --- | --- |
| Default route | `journey-starter-model` |

## Intended use

Use this example when learning `vllm-sr recipe scaffold`, journey validation,
or review-bundle workflows. Copy patterns into a maintained recipe directory
when you are ready to promote the profile.

## Routing behavior

All requests route to the default catch-all decision.

## Requirements

- Reachable OpenAI-compatible endpoint at `host.docker.internal:8000`.
- Replace placeholder endpoints before activation.
- Pass secrets with `vllm-sr serve --recipe-env VAR` when env-backed credentials are configured.

## Data handling and safety

Review data retention, replay, and plugin behavior before production use.

## Quick start

```bash
vllm-sr validate --config config/recipes/examples/journey-starter/config.yaml
python3 tools/agent/scripts/vllm_sr_journey.py validate \
  --config config/recipes/examples/journey-starter/config.yaml \
  --recipe-dir config/recipes/examples/journey-starter
vllm-sr serve --config config/recipes/examples/journey-starter/config.yaml
```

## Evaluation

Starter probes live in [`probes.yaml`](probes.yaml). See
[`../../CONFORMANCE.md`](../../CONFORMANCE.md) for the maintained recipe contract.

## Limitations

- Placeholder backends are not production-ready.
- Probe coverage is minimal until expanded.
- This example is not part of the maintained catalog matrix.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Contributor journey tutorial](https://github.com/vllm-project/semantic-router/blob/main/website/docs/tutorials/agent/vllm-sr-journey.md)
