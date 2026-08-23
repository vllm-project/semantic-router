---
title: Container Connectivity
sidebar_label: Container Connectivity
---

# Container Connectivity

Use this guide when the local stack starts but cannot reach a model backend, or
when the host cannot reach the Router, Envoy, Dashboard, or metrics endpoints.

## Start with the failing hop

A routed request crosses several network boundaries:

```text
client -> Envoy -> Router -> selected provider backend
```

Check each hop in that order:

```bash
vllm-sr status
vllm-sr logs envoy
vllm-sr logs router
curl -sS http://localhost:8899/v1/models
```

If `/v1/models` is unavailable, focus on the local stack and published ports.
If it succeeds but completions fail, inspect the selected provider name in the
response/logs and test that provider from the Router's network.

## `localhost` means the current container

A common configuration error is pointing a provider at `localhost` even though
the inference server runs on the host or in another container. From the Router
container, `127.0.0.1` and `localhost` refer to the Router container itself.

Configure an address that is reachable from the runtime network:

```yaml
models:
  - name: local-model
    card: {}
    connections:
      - provider: vllm
        endpoint: http://model-server:8000/v1
        model: local-model
```

Use a container DNS name when both services share a network. For a model server
running directly on the host, use the container runtime's supported host
gateway name or a host IP that the container can reach. On Kubernetes, use a
Service DNS name rather than a pod IP.

## Make the backend listen beyond loopback

The model server must bind to an interface reachable by its clients. For
example, a host-side vLLM server generally needs `--host 0.0.0.0`:

```bash
vllm serve <model-id> \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name local-model
```

Binding to `0.0.0.0` does not provide authentication. Restrict the port with a
firewall or private network, and use the provider's authentication support when
traffic crosses a trust boundary.

## Test from the same network namespace

A successful host-side request proves only host reachability. Test from a
temporary container on the same runtime network, or from the Router container
using your container runtime's inspection tools. Query the backend's
OpenAI-compatible model endpoint:

```bash
curl -sS http://<reachable-backend>:8000/v1/models
```

For Kubernetes, start with the Service and endpoints:

```bash
kubectl get service,endpoints -n <namespace>
kubectl run network-check \
  --rm -i --restart=Never \
  --image=curlimages/curl \
  -n <namespace> -- \
  curl -sS http://<service-name>:8000/v1/models
```

Remove or restrict temporary diagnostic pods according to your cluster policy.

## Check firewalls and security rules

If the host can reach a backend but the runtime cannot, inspect:

- host firewall rules for the backend port;
- cloud security groups or network ACLs;
- Kubernetes NetworkPolicies;
- corporate proxies that intercept only some address ranges; and
- DNS resolution inside the container or pod.

Open only the source networks and ports the deployment needs. Avoid making a
provider endpoint public merely to diagnose an internal route.

## Published local ports

The default local stack publishes these user-facing endpoints:

| Endpoint | Default address |
|----------|-----------------|
| OpenAI-compatible listener through Envoy | `http://localhost:8899` |
| Dashboard | `http://localhost:8700` |
| Router management API | `http://localhost:8080` |
| Router metrics | `http://localhost:9190/metrics` |

Management endpoints may require authentication, depending on the active
configuration. A nonzero `VLLM_SR_PORT_OFFSET` shifts every published host port;
use `vllm-sr status` and `vllm-sr dashboard` rather than assuming defaults.

If a port is already occupied, either stop the conflicting process or run an
isolated stack:

```bash
VLLM_SR_STACK_NAME=lane-b \
VLLM_SR_PORT_OFFSET=200 \
vllm-sr serve
```

Use the same two environment variables with `status`, `logs`, `dashboard`, and
`stop` for that stack.

## Grafana shows no data

Grafana and Prometheus are present only when observability is enabled. Check the
metrics source before debugging panels:

```bash
curl -sS http://localhost:9190/metrics | head
```

Then confirm that Prometheus can scrape the Router and Envoy targets. Empty
panels can be correct when no request has exercised the corresponding path; for
example, rejection and cache metrics remain empty until a policy rejects or a
cache handles a request.

Also verify:

- the selected time range includes recent traffic;
- the active stack's port offset is reflected in local URLs;
- histogram panels query bucket metrics with an appropriate rate window; and
- labels in a copied dashboard match the metrics emitted by this version.

## Quick checklist

- The local stack is running and `vllm-sr status` identifies the failing
  component.
- Envoy's `/v1/models` endpoint is reachable from the client.
- The provider endpoint is not `localhost` unless it truly runs in the same
  container.
- The backend listens on a reachable interface and exposes `/v1/models`.
- DNS, firewall, security-group, and NetworkPolicy rules permit the required
  path.
- Model names in Router config match the names served by the provider.
- Metrics are emitted before investigating Grafana queries.

For registry or artifact-download failures, see
[Restricted Network Environments](./network-tips).
