---
title: Inference Platforms
description: Place semantic model selection above a platform that owns model deployment and replica scheduling.
---

# Inference Platforms

Inference platforms and Semantic Router solve different routing problems.
Semantic Router chooses a model or model pool from request meaning and policy.
The inference platform deploys that model and chooses a replica according to
capacity, locality, and health.

## Choose an integration

| Platform | Start with | Typical ownership |
| --- | --- | --- |
| vLLM Production Stack | [Production Stack](production-stack) | vLLM model services, discovery, and replica routing. |
| AIBrix | [AIBrix](aibrix) | Model deployment, autoscaling, and replica-level traffic management. |
| llm-d | [llm-d](llm-d) | `InferencePool` endpoint selection and distributed inference patterns. |
| NVIDIA Dynamo | [Dynamo](dynamo) | Dynamo graphs, workers, and frontend lifecycle. |

Use the platform your infrastructure team already supports. These guides do
not replace the platform's release-specific installation, sizing, or upgrade
documentation.

## Keep the two layers aligned

For every model pool, align:

- the Semantic Router provider name and `provider_model_id`;
- the platform's served model identity;
- the stable Service, Gateway, or frontend address; and
- the modality, context, tool, and protocol capabilities declared by routing
  policy.

Use Service DNS or a managed gateway address rather than a Kubernetes
`ClusterIP`. Keep replica scheduling out of semantic policy, and keep prompt
meaning out of the replica scheduler.

## Validate the complete path

1. Verify direct generation from every model pool.
2. Validate the canonical Router configuration.
3. Verify a request through each public virtual model.
4. Confirm both the semantic selection and the serving replica.
5. Exercise unavailable-model behavior explicitly; do not assume the Router or
   platform provides cross-model fallback unless you configured and tested it.

For the surrounding deployment choices, return to
[Choose a Deployment](../deployment-options).
