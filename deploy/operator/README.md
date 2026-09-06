# Semantic Router Operator

The Operator reconciles a `SemanticRouter` custom resource into Router
configuration, workload, Service, storage, autoscaling, and optional
platform-specific resources. Use it when Kubernetes should own the Router
lifecycle rather than a local CLI or a standalone Helm release.

User-facing installation and day-two operations are documented in the
[Operator guide](../../website/docs/installation/k8s/operator.md) and
[operations guide](../../website/docs/installation/k8s/operator-operations.md).
The generated field reference is
[`website/docs/api/semantic-router-crd.md`](../../website/docs/api/semantic-router-crd.md).
This README is for contributors working on the Operator source.

## Install from this checkout

Requirements:

- a Kubernetes cluster and matching `kubectl` context;
- Go and the container toolchain required by the Makefile;
- permission to install CRDs and cluster-scoped RBAC.

```bash
cd deploy/operator

# Install or update the CRD
make install

# Build and publish the controller image, then deploy it
make docker-build docker-push \
  IMG=registry.example.com/semantic-router-operator:dev
make deploy IMG=registry.example.com/semantic-router-operator:dev
```

Inspect the controller and CRD before creating a Router:

```bash
kubectl get deployments --all-namespaces \
  -l app.kubernetes.io/name=semantic-router-operator
kubectl explain semanticrouter.spec
```

## Choose a sample

The samples are executable CRs, not drop-in production values. Review image
tags, backend names, namespaces, storage classes, credentials, resources, and
model paths before applying one.

| Sample | What it demonstrates |
| --- | --- |
| `vllm.ai_v1alpha1_semanticrouter_simple.yaml` | Small standalone CR with a KServe backend. |
| `vllm.ai_v1alpha1_semanticrouter_gateway.yaml` | Existing-Gateway mode; the user still owns the `HTTPRoute`. |
| `vllm.ai_v1alpha1_semanticrouter_llamastack.yaml` | Label-based Llama Stack service discovery. |
| `vllm.ai_v1alpha1_semanticrouter_openshift.yaml` | OpenShift-oriented workload and Route settings. |
| `vllm.ai_v1alpha1_semanticrouter_route.yaml` | OpenShift Route creation. |
| `vllm.ai_v1alpha1_semanticrouter_{redis,valkey,milvus,qdrant}_cache.yaml` | External response-cache backend configuration. |
| `vllm.ai_v1alpha1_semanticrouter_hybrid_cache.yaml` | In-memory HNSW with persistent cache storage. |
| `vllm.ai_v1alpha1_semanticrouter_mmbert.yaml` | mmBERT embedding configuration. |
| `vllm.ai_v1alpha1_semanticrouter_complexity.yaml` | Complexity signals and conditional routing. |

For example:

```bash
kubectl apply -f \
  deploy/operator/config/samples/vllm.ai_v1alpha1_semanticrouter_simple.yaml
kubectl get semanticrouters
kubectl describe semanticrouter semantic-router-simple
```

The Operator configures external Redis, Valkey, Milvus, Qdrant, KServe, Llama
Stack, and model-serving resources; it does not create those systems unless a
specific reconciler explicitly says otherwise.

## Deployment modes

### Standalone

Without `spec.gateway.existingRef`, the reconciled pod includes the local Envoy
path used to send requests through the Router's ExtProc service. This is the
self-contained mode for a cluster without a shared Gateway.

### Existing Gateway

With `spec.gateway.existingRef`, the controller resolves the referenced Gateway
and changes the workload to gateway-integration mode. Automatic `HTTPRoute`
creation is not implemented. Apply and manage a route that matches your Gateway
and Service separately.

Do not assume the Gateway sample creates ingress traffic just because the CR is
ready; verify both the Gateway reference and your `HTTPRoute` status.

### OpenShift Route

`spec.openshift.routes.enabled` asks the controller to create an OpenShift
`Route`. Choose `edge`, `passthrough`, or `reencrypt` termination to match the
backend TLS contract. A passthrough or re-encrypt route requires the backend
side of that contract to be configured as well.

## Backend discovery

Each `spec.vllmEndpoints` entry uses one backend type:

| Type | Resolution |
| --- | --- |
| `kserve` | Predictor Service created for an `InferenceService` in the CR namespace. |
| `llamastack` | Service selected by the configured Kubernetes labels. |
| `service` | Explicit Service name, namespace, and port. |

After changing discovery behavior, test missing, ambiguous, and cross-namespace
references as well as the successful path.

## Develop and validate

```bash
cd deploy/operator

# Run the controller against the current kubeconfig
make run

# Unit/envtest checks plus generated-code prerequisites
make test

# Regenerate CRD/RBAC and deepcopy code after API changes
make manifests generate
```

From the repository root, use the canonical generated-artifact and integration
checks:

```bash
make generate-crd
(cd deploy/operator && make test)
```

After an API-field change, keep these surfaces aligned:

1. `api/v1alpha1/semanticrouter_types.go`;
2. admission validation and tests;
3. controller translation and tests;
4. CRD bases and bundle manifests;
5. affected samples;
6. the generated website CRD reference.

Do not edit generated CRDs or the generated reference as the only source of a
field change.

## Observe a reconciliation

```bash
kubectl get semanticrouters
kubectl describe semanticrouter <name>
kubectl get events --sort-by=.lastTimestamp
kubectl logs --all-containers \
  -l app.kubernetes.io/name=semantic-router-operator \
  --all-namespaces
```

Read status conditions before inspecting generated pods. They distinguish an
invalid CR, unresolved backend or Gateway, storage failure, and rollout
failure more clearly than pod readiness alone.

## Remove a development install

Delete test `SemanticRouter` resources before removing the controller. Remove
the CRD only when no retained custom resources are needed:

```bash
cd deploy/operator
make undeploy
make uninstall
```

PVC and external cache data have independent retention behavior; inspect them
before cleanup.
