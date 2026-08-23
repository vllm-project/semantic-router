# Dynamo integration assets

This directory contains the Semantic Router gateway resources and an older,
pinned Dynamo model-deployment example.

Start with the current
[Dynamo integration guide](../../../../website/docs/installation/k8s/dynamo.md).
Dynamo APIs and runtime images evolve independently from Semantic Router; that
guide separates the current Dynamo installation from the stable Router and
Envoy Gateway wiring.

## Compatibility boundary

[`dynamo-graph-deployment.yaml`](dynamo-graph-deployment.yaml) and the sibling
[`helm-chart/`](../helm-chart/) use the older `nvidia.com/v1alpha1`
`DynamoGraphDeployment` API and runtime tag `0.6.1.post1`. Do not apply them to
a current Dynamo platform unless its CRD and runtime compatibility explicitly
match those versions.

The old example also assumes:

- NVIDIA device resources and driver libraries at a host-specific path;
- privileged containers;
- three GPU assignments for the frontend, prefill worker, and decode worker;
- ETCD and NATS Services in `dynamo-system`;
- TinyLlama as the default worker model.

These assumptions make it a historical integration fixture, not a generic or
production deployment profile.

## Files still used by the current integration

| File | Purpose |
| --- | --- |
| [`gwapi-resources.yaml`](gwapi-resources.yaml) | Gateway, `HTTPRoute`, cross-namespace permission, and ExtProc `EnvoyPatchPolicy`. |
| [`envoy-gateway-values.yaml`](envoy-gateway-values.yaml) | Enables the Envoy Gateway extension required by the patch policy. |
| [`rbac.yaml`](rbac.yaml) | Access used by the Router integration where required. |

Before applying `gwapi-resources.yaml`, update:

- the Dynamo frontend Service name, namespace, port, and served-model name;
- the Semantic Router Service address;
- Gateway and route namespaces;
- the `ReferenceGrant` for every cross-namespace backend reference.

Render and inspect environment-specific values rather than copying a Service
name or address from a previous cluster.

## Request ownership

```text
client
  -> Envoy Gateway
     -> Semantic Router ExtProc policy and model selection
        -> HTTPRoute
           -> Dynamo frontend
              -> Dynamo worker selection and KV-cache routing
```

Semantic response caching and Dynamo KV-cache routing are different layers. A
Router cache may reuse an entire prior response; Dynamo uses token-prefix state
while serving a new request.

## Verify an integration

First prove the Dynamo frontend works directly. Then inspect Gateway and patch
policy status before sending the same request through Envoy:

```bash
kubectl get services --namespace dynamo-system
kubectl get gateway,httproute --all-namespaces
kubectl describe envoypatchpolicy semantic-router-extproc-patch-policy \
  --namespace default
```

Use the frontend's `/v1/models` response as the model-name source of truth. The
same name must appear in a canonical Semantic Router Model and the Gateway route.

If this repository's pinned 0.6.1 fixture is intentionally under test, record
that version in the result. Do not report it as validation of a newer Dynamo
release.
