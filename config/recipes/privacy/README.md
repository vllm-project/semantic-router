# Privacy-First Routing Recipe Model Card

## Overview

Privacy-First Routing keeps sensitive and suspicious requests on a local model.
Only clearly non-sensitive requests that need deeper reasoning may use the
configured frontier lane; ordinary requests also default to local processing.

This policy reduces accidental cloud exposure, but it cannot by itself prove
that a backend, network, or storage system is private.

## Model details

| Lane | Reference alias | Purpose |
| --- | --- | --- |
| Local | `local/private-qwen` | Default, sensitive-data, and attack-containment traffic. |
| Frontier | `cloud/frontier-reasoning` | Explicitly non-sensitive, high-effort reasoning. |

Clients use `vllm-sr/auto`. Replace both aliases with backends that match the
deployment's data-handling policy.

## Intended use

Use this recipe for:

- prompts containing personal, confidential, or proprietary information;
- deployments with a local-only handling requirement;
- suspicious prompts that should not receive tools or leave the local boundary;
- mixed workloads where a frontier model is allowed only for non-sensitive
  reasoning; and
- teams that need a local fallback for every unmatched request.

Do not use it as the sole control for regulated data. Network isolation,
backend ownership, logging, retention, and access controls remain necessary.

## Routing behavior

| Request pattern | Route | Behavior |
| --- | --- | --- |
| Jailbreak or attack signal | `local_security_containment` | Keep the request local with containment policy. |
| PII, private-context, or local-only signal | `local_privacy_policy` | Keep the request local with privacy-oriented reasoning. |
| Clearly non-sensitive deep reasoning | `cloud_frontier_reasoning` | Use the configured frontier backend. |
| Everything else | `local_standard` | Use the local model. |

Security and privacy decisions outrank the frontier route. The frontier route
requires positive reasoning evidence and exclusion of privacy or attack
signals.

## Requirements

- A local OpenAI-compatible backend controlled by the operator.
- A frontier backend only if policy permits it.
- Jailbreak, PII, privacy-KB, embedding, context, structure, and complexity
  signals configured in [`config.yaml`](config.yaml).
- Logging and replay settings that match the deployment's retention policy.

## Data handling and safety

Routing happens before provider invocation. Requests selected for local lanes
still pass through the Router and any enabled supporting stores or logs. Review
those components before claiming end-to-end local processing. Treat classifier
errors as possible: conservative fallbacks reduce risk but do not eliminate it.

All four checked-in routes enable in-memory Router Replay. The effective replay
defaults capture up to 2 KiB each from the request and response body, including
the local security-containment and privacy routes. Local model placement does
not make that replay data harmless. Disable body capture or replay when prompts
and responses must not be retained, and apply the same access controls to the
replay store as to the original traffic.

## Quick start

```bash
vllm-sr validate --config config/recipes/privacy/config.yaml
vllm-sr serve --config config/recipes/privacy/config.yaml
```

## Evaluation

The probes cover PII, local-only requests, jailbreak and attack prompts,
non-sensitive reasoning, multilingual variants, priority collisions, and the
local fallback. See [`probes.yaml`](probes.yaml) and the
[conformance guide](../CONFORMANCE.md).

## Limitations

- Classifiers may miss sensitive content or produce false positives.
- A local alias is private only when its endpoint, network, storage, and logs
  are also controlled appropriately.
- The recipe does not redact captured request or response bodies by default.
- The frontier route should be removed entirely in environments that prohibit
  all external inference.

## References

- [Recipe metadata](metadata.yaml)
- [Runtime configuration](config.yaml)
- [Routing DSL](recipe.dsl)
- [Evaluation probes](probes.yaml)
- [Recipe authoring and conformance](../CONFORMANCE.md)
