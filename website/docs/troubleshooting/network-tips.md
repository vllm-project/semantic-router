---
title: Restricted Network Environments
sidebar_label: Restricted Networks
---

# Restricted Network Environments

Semantic Router may need network access for three different reasons:

1. the container runtime pulls Router, Dashboard, Envoy, and supporting images;
2. the Router downloads classifier or embedding artifacts; and
3. routed requests call your configured model providers.

Identify which layer is failing before changing mirrors or proxy settings. A
registry timeout, a Hugging Face timeout, and an unreachable provider endpoint
need different fixes.

## Diagnose the failing layer

Start the stack and inspect its status and component logs:

```bash
vllm-sr serve
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy
```

| Symptom | Likely layer |
|---------|--------------|
| Image pull or registry authentication error | Container registry |
| Router starts but waits while loading a model artifact | Hugging Face or local model path |
| Router and Envoy are ready, but completions return connection errors | Provider endpoint or firewall |
| Kubernetes pod remains in `ImagePullBackOff` | Cluster node registry access |

## Container images

Pre-pull the images from a network that can reach their registries, or mirror
them into a registry available to the deployment environment. For local
development, confirm that every required image exists before using a no-pull
policy:

```bash
vllm-sr serve --image-pull-policy never
```

`never` does not download missing images; startup fails if an image is absent.
Use `ifnotpresent` when local images should be reused but missing ones may still
be pulled.

When you build the project from source, configure the package manager and
container runtime through your organization's approved proxy or mirror. Avoid
checking regional endpoints, credentials, or local proxy addresses into the
repository.

## Hugging Face downloads

The local CLI forwards `HF_ENDPOINT`, `HF_TOKEN`, `HF_HOME`, and
`HF_HUB_CACHE` to the Router container. Set only the values required by your
environment:

```bash
export HF_ENDPOINT=https://your-approved-hugging-face-mirror.example
export HF_TOKEN=your_token_if_required
vllm-sr serve
```

Keep tokens in the environment or an external secret manager. The CLI masks
sensitive passthrough values in its logs.

For an offline deployment, download the required artifacts in advance and put
them in the workspace `models/` directory, which is mounted at `/app/models`.

Use `/app/models/...` in Router configuration, then verify that the file exists
inside the runtime and that its format matches the selected signal or
embedding implementation.

## Provider endpoints

Provider URLs must be reachable from the Router/Envoy network, not only from
the host shell. Do not use `localhost` for a backend running in another
container or on the host; inside a container, `localhost` refers to that
container itself.

Use one of these patterns:

- a service name on the same container network;
- a host address or host-gateway name reachable from the container runtime;
- a Kubernetes Service DNS name; or
- a routable private or public provider endpoint.

See [Container Connectivity](./container-connectivity) for a step-by-step
endpoint and firewall checklist.

## Kubernetes image pulls

Kubernetes nodes use their own container runtime and do not inherit image cache
or proxy settings from the machine running `kubectl`.

For restricted clusters:

- mirror required images into a registry reachable by every node;
- configure `imagePullSecrets` for authenticated registries;
- use an appropriate pull policy after images are present;
- for development clusters, preload images with the cluster tool's supported
  command; and
- inspect pod events to distinguish DNS, authentication, rate-limit, and
  missing-image errors.

```bash
kubectl describe pod <pod-name> -n <namespace>
kubectl get events -n <namespace> --sort-by=.lastTimestamp
```

## What not to do

- Do not commit API tokens, registry credentials, proxy passwords, or private
  mirror addresses.
- Do not disable TLS verification as a permanent workaround.
- Do not assume a successful host-side `curl` proves container or pod
  reachability.
- Do not replace checked-in Dockerfiles with environment-specific copies;
  carry organization-specific build configuration outside the source tree.

## Related guides

- [Container Connectivity](./container-connectivity)
- [Security Hardening](../installation/security-hardening)
- [Quickstart](/docs/installation)
