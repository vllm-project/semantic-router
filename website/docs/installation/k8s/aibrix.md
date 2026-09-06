---
title: Integrate with AIBrix
description: Combine Semantic Router model-pool selection with AIBrix model deployment and replica routing.
---

# Integrate with AIBrix

Use this topology when AIBrix owns model deployment, scaling, and replica-level
routing, while Semantic Router should select a model pool from request meaning
and policy.

AIBrix releases independently from Semantic Router. Install it with the
[current AIBrix installation guide](https://aibrix.readthedocs.io/latest/getting_started/installation/installation.html)
and use its maintained
[Semantic Router sample](https://github.com/vllm-project/aibrix/tree/main/samples/semantic-router)
for release-specific manifests. This page explains how the two control layers
fit together.

## Responsibility split

| Component | Owns |
| --- | --- |
| Semantic Router | Signals, decisions, model-pool selection, and recipe-scoped plugins. |
| AIBrix | Model workloads, autoscaling, service discovery, and replica-level routing. |
| Gateway API | Client traffic and the ExtProc call to Semantic Router. |

Semantic Router selects the logical model. AIBrix then chooses a healthy
replica serving that model. Keep autoscaling and replica load policy in AIBrix;
do not duplicate it in semantic decisions.

## Before you begin

You need:

- a Kubernetes cluster supported by the AIBrix release you selected;
- Gateway API and the gateway required by that release;
- `kubectl`, Helm, model credentials, and sufficient inference capacity; and
- at least two model names if you want to observe semantic model selection.

Pin the AIBrix release and inspect its release notes before applying manifests.
Do not copy a historical release URL from this page or assume a development tag
is suitable for production.

## 1. Deploy and verify AIBrix

Follow the upstream installation guide and deploy the model services. Before
adding Semantic Router, verify each served model through the AIBrix gateway:

```bash
kubectl get pods -A
kubectl get services -A
```

Send a direct Chat Completions request for every model name you plan to expose.
This confirms model access, capacity, and gateway routing independently of the
semantic policy.

## 2. Align model identities

For each AIBrix model, create a Semantic Router provider model whose
`provider_model_id` is the name accepted by the AIBrix endpoint. Bind the
backend to the stable Gateway or Service DNS name, not to a `ClusterIP`.

Then reference those provider names from model cards and decisions. The same
identity must agree in four places:

1. the request-facing virtual model or entrypoint;
2. the Semantic Router provider model;
3. the AIBrix/Gateway routing resource; and
4. the model server's served model name.

Validate the complete Router document before deployment:

```bash
vllm-sr validate --config config.yaml
```

## 3. Deploy Semantic Router and the Gateway policy

The upstream sample contains a complete integration for its current AIBrix
release. If you adapt it instead of using the sample unchanged:

- deploy the Router config with `configOverride` as described in
  [Configuration Workflows](../configuration-workflows#helm);
- keep provider credentials in Kubernetes Secrets;
- preserve the ExtProc processing mode expected by the gateway; and
- update Router providers and Gateway backends together.

For large request bodies or immediate streamed responses, review
[Streamed ExtProc](streamed-extproc) before changing the processing mode.

## 4. Verify the integration

1. Send a direct request to each AIBrix model.
2. Send requests through a Semantic Router virtual model.
3. Inspect the Router selection headers.
4. Confirm the selected AIBrix model and serving replica.

Use [Test a Kubernetes Gateway Deployment](gateway-testing) for the common
checks. A correct route decision and a successful generation are separate
signals; test both.

## Cleanup

Remove the Router and gateway resources with the release names you installed.
Remove AIBrix using its versioned uninstall procedure. Model volumes and caches
may outlive Deployments, so inspect them before deleting persistent data.
