# OpenShift deployment example

This directory adapts Semantic Router's Kubernetes resources to OpenShift
Routes, security constraints, ImageStreams/builds, and optional KServe
backends. The deployment script discovers Service addresses and generates the
Router ConfigMap for the selected namespace.

The assets are an integration example. Review images, credentials, Route TLS,
storage, resources, and cluster-wide operator installation before using them
in a shared environment.

## Choose a backend mode

| Command option | Backend |
| --- | --- |
| `--kserve --simulator` | KServe simulator resources. |
| `--kserve` | A real KServe model; requires its GPU and model-serving prerequisites. |
| no backend flag | CPU provider-mocker backends from a qualified image digest. |

`--classifier-gpu` moves the Router classifier to a GPU independently of the
backend choice. `--no-observability` skips Dashboard, Grafana, Prometheus, and
the other optional UI resources.

## Deploy

Log in with `oc`, select a project name, and preview the script options:

```bash
oc whoami
deploy/openshift/deploy-to-openshift.sh --help
```

For a CPU-only disposable project, supply the published fixture digest:

```bash
export PROVIDER_MOCKER_IMAGE="ghcr.io/vllm-project/semantic-router/provider-mocker@sha256:<qualified-digest>"
deploy/openshift/deploy-to-openshift.sh \
  --namespace vllm-semantic-router-system
```

For the core path without the optional observability UIs:

```bash
deploy/openshift/deploy-to-openshift.sh \
  --namespace vllm-semantic-router-system \
  --no-observability
```

The default fixture does not perform model inference. For a local real-model
check, use the optional [Qwen3-0.6B runner](../../tools/test/services/tiny-model/README.md).

The KServe modes call the helpers in [`../kserve/`](../kserve/). Install and
validate KServe before using a real model. Do not install cluster operators
from an application script when the platform team owns them.

## What the script changes

Depending on the selected mode, the script:

1. creates or selects the namespace;
2. creates model or simulator resources;
3. discovers backend Service addresses;
4. renders `config-openshift.yaml` into a temporary canonical Router config;
5. creates Router and Envoy ConfigMaps;
6. applies the Router, Envoy, Services, and Routes;
7. optionally builds and deploys Dashboard and observability resources.

The dynamic address flow is described in
[`README-DYNAMIC-IPS.md`](README-DYNAMIC-IPS.md). The checked-in config contains
placeholders and should not be edited with cluster IPs.

## Grafana Live origins

When an external proxy forwards an internal `Host`, Grafana Live can reject
`/embedded/grafana/api/live/ws` even if panels load. Supply the trusted public
**Dashboard origin**, not the internal Grafana service URL or Grafana Route:

```bash
export GF_LIVE_ALLOWED_ORIGINS="https://dashboard.example.com,https://dashboard.example.net:8443"
deploy/openshift/deploy-to-openshift.sh --namespace vllm-semantic-router-system
```

The script passes this value to the Grafana container without changing
`GF_SERVER_ROOT_URL`. Unset leaves the manifest's value intact; an explicit
empty value clears an override and preserves Grafana's default origin checks.
Values are comma-separated origins (scheme, hostname, optional port), with no
path or trailing slash. Do not use `*` to bypass validation.

The script does not infer allowed origins: the Dashboard Route is created
after Grafana, and another proxy may expose a different public hostname. If
using the generated Route directly, discover its host after deployment with
`oc get route dashboard -n <namespace> -o jsonpath='{.spec.host}'`. Combine it
with the Route's external scheme (`https` for the checked-in edge-TLS Route),
then supply the resulting origin on the next deployment. For an existing
installation, update only Grafana instead of rebuilding the whole stack:

```bash
oc set env deployment/grafana --containers=grafana \
  --namespace vllm-semantic-router-system \
  GF_LIVE_ALLOWED_ORIGINS="$GF_LIVE_ALLOWED_ORIGINS"
```

If applying static manifests without the script, patch the same environment
variable in `observability/grafana/deployment.yaml`, alongside the existing
root-URL and namespace substitutions. Shell environment variables alone do
not modify static YAML. Keep the origin in your deployment source so a later
apply does not reset it. After rollout, verify an allowed origin connects to
Live and an unrelated origin is rejected; authentication and WebSocket proxy
configuration are still required.

## Verify

```bash
oc get pods,services,routes \
  --namespace vllm-semantic-router-system
oc logs deployment/semantic-router \
  --namespace vllm-semantic-router-system
```

Discover Route hosts instead of copying an address from another cluster:

```bash
ENVOY_HOST="$(oc get route envoy-http \
  --namespace vllm-semantic-router-system \
  -o jsonpath='{.spec.host}')"

curl --fail-with-body "http://$ENVOY_HOST/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 32
  }'
```

The Route name and scheme can differ when you patch the example. Use
`oc get routes` as the source of truth.

For a repeatable scenario check, run
[`demo-routing-test.sh`](demo-routing-test.sh). The helpers under [`demo/`](demo/)
are presentation and diagnostics tools; they are not accuracy or performance
benchmarks.

## OpenShift-specific boundaries

- Workloads must satisfy the namespace's SCC; avoid granting `anyuid` or
  privileged access unless the image genuinely requires it.
- Routes need an explicit TLS termination policy for external use.
- Binary builds send the current checkout to the cluster build service. Review
  the build context and resulting ImageStream before use.
- Model and provider credentials belong in Secrets, not ConfigMaps or shell
  output.
- GPU enablement belongs to the cluster's supported accelerator operator and
  node configuration, not to this application example.

## Diagnose failures

```bash
oc get events --namespace vllm-semantic-router-system \
  --sort-by=.lastTimestamp
oc describe deployment/semantic-router \
  --namespace vllm-semantic-router-system
oc logs deployment/envoy \
  --namespace vllm-semantic-router-system
```

| Symptom | First check |
| --- | --- |
| Router ConfigMap has unusable endpoints | Model Services and the dynamic config render. |
| Pod is rejected | SCC events and container security context. |
| Route is unavailable | Route admission, Service endpoints, and TLS termination. |
| Model never becomes ready | Backend logs, GPU scheduling, PVC, egress, and model credentials. |
| Dashboard build is stale | Latest BuildConfig run, ImageStream tag, and deployment rollout. |

## Cleanup

Preview cleanup before deleting a namespace or storage:

```bash
deploy/openshift/cleanup-openshift.sh \
  --namespace vllm-semantic-router-system \
  --dry-run
```

Then choose the narrowest cleanup level shown by `--help`. Shared operators,
images, model caches, and persistent data are outside the example's ownership
unless you created them solely for this deployment.
