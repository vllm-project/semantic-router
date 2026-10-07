---
title: Gateway Modes
description: Choose where client traffic enters the Router, standalone or behind an Envoy-based gateway, on the docker and kubernetes targets.
---

# Gateway Modes

`vllm-sr serve` makes two choices: the **gateway mode**, which is where client
traffic enters, and the **target**, which is where the stack runs.

| Mode | Client traffic enters at | Use it for |
| --- | --- | --- |
| `standalone` (default) | the Router, which serves the OpenAI-compatible API on the config's `listeners` | one host, development, edge, most self-hosting |
| `extproc` | an Envoy-based gateway in front of the Router, which serves ext_proc | Envoy features such as rate limiting, mTLS and advanced route matching, or a gateway you already run |

| Target | `standalone` | `extproc` |
| --- | --- | --- |
| `docker` (default) | the Router container serves the listeners; there is no Envoy container | the Envoy container the CLI starts, in front of the Router, as in earlier releases |
| `kubernetes` | the Router pods serve the listeners, and the Service exposes them | the Router serves ext_proc for your Envoy Gateway, Envoy AI Gateway, Istio or KServe |

```bash
vllm-sr serve                                      # standalone, docker
vllm-sr serve --gateway extproc                    # Envoy in front, docker
vllm-sr serve --target kubernetes --config config.yaml
```

Start with `standalone` unless you need one of the Envoy features above. On
docker, switching is a restart on the active configuration: run
`vllm-sr serve --gateway extproc`, and plain `vllm-sr serve` to come back. In
standalone mode there is no Envoy container, so `vllm-sr logs envoy` and the
`x-envoy-*` response headers belong to `extproc` only.

Both modes run the same routing core, so a request gets the same decision, the
same upstream request and the same response transformations in either. The
[design](../proposals/standalone-mode#differences-from-envoy-mode) lists the few
deliberate differences, such as the `x-envoy-*` headers that only Envoy adds.

:::note Upgrading
Earlier releases always put Envoy in front of the Router. Standalone is now the
default; `vllm-sr serve --gateway extproc` restores the previous stack exactly.
See the [release note](../release-notes/standalone-mode).
:::

## What a standalone Router does

- **Listeners:** each entry of `listeners` is served with HTTP/1.1 and HTTP/2
  (h2c in cleartext), with its `timeout` as the idle timeout, a 500 MiB request
  body limit and at most 50,000 connections, the limits of the Envoy template.
- **API keys:** with `api_keys` set, a client sends one of them as
  `Authorization: Bearer <key>` or `api-key: <key>`; other requests get an
  OpenAI-style 401. The key is removed before the request reaches a provider.
- **Model allow-list:** with `models` set, the listener accepts only those
  request models (see [Model allow-list](#model-allow-list)).
- **TLS:** `tls` serves the listener over TLS 1.2 or later, with HTTP/2 or
  HTTP/1.1 negotiated by ALPN. Relative paths are relative to the config file's
  directory. The Router reloads the key pair when its files change, so a renewed
  certificate (a rotated Kubernetes secret, cert-manager) serves new connections
  without a restart; a pair that fails to load leaves the previous one serving.
  `--gateway extproc` does not serve it.

  ```yaml
  listeners:
    - name: https-8443
      address: 0.0.0.0
      port: 8443
      tls:
        cert_file: certs/tls.crt
        key_file: certs/tls.key
  ```

- **The edge's trust boundary:** client-sent identity headers (`x-authz-*`, and
  the names `global.services.authz.identity` sets), unless the listener trusts
  them (see [Identity headers](#identity-headers)), and the proxy-control
  headers only a trusted proxy may set never reach routing or a backend. Those are
  `x-envoy-internal` and Envoy's retry, timeout and tracing controls, such as
  `x-envoy-max-retries`, `x-envoy-retry-on` and `x-envoy-upstream-rq-timeout-ms`.
  Envoy-based layers behind the Router (sidecars, gateways in front of model
  servers) obey them from a caller they trust, which the Router is, so a client
  could otherwise set retries and timeouts there; the Router's reliability
  policy stays the one retry and timeout authority. The list is the one Envoy
  strips from external requests, and it stays that narrow.
- **Probes:** `GET /health` answers while the process runs, and `GET /ready` once
  the routing core can take traffic. Prometheus metrics stay on the Router's
  metrics port (9190).
- **Reloads:** the Router reloads its config in place. A change to a listener's
  address, port, timeout or `tls` paths, or a new or removed listener, is
  rejected as `restart_required` until the Router restarts; API keys and model
  allow-lists reload in place.

### Model allow-list

A listener's `models` lists the only request `model` values it accepts. Use it
to give a public key access to the router's auto model and nothing else, while
an internal listener keeps every model:

```yaml
listeners:
  - name: dashboard-internal   # first listener: the Dashboard Playground uses it
    address: 127.0.0.1
    port: 8898
  - name: public
    address: 0.0.0.0
    port: 8899
    api_keys: ["${WORKSHOP_KEY}"]
    models: [vllm-sr/auto]
```

- Names match exactly (case-sensitive, after trimming the request value), and
  aliases are not expanded: list every name clients may send. Empty or absent,
  the listener accepts every model.
- The check runs after the API key check and before any signal, cache or
  decision, on the model the Router parsed for routing. Any other model,
  including a provider model that would otherwise pass through, gets
  `403` with `{"error": {"code": "model_not_allowed", ...}}` in the client's
  protocol. A request without a model gets `400 model_required`.
- `GET /v1/models` on the listener lists only the allowed names the catalog
  has.
- The model calls a decision makes in process (Looper and request-graph hops)
  are not client requests and are not restricted, so `vllm-sr/auto` can still
  reach every provider model its decisions name.
- The listener ignores the `x-vsr-skip-processing` opt-out even when
  `global.router.skip_processing.enabled` is on, because a skipped request
  would bypass the check.
- `--gateway extproc` rejects a listener with `models` as unsupported: the
  Envoy listener the CLI generates does not enforce it yet.

### Identity headers

By default a standalone listener drops the identity headers a client sends,
because nothing in front of the Router has authenticated it: anyone could claim
any user. Behind an authenticating proxy, or for a trusted application server
that sets the user itself, let the listener keep them:

```yaml
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    identity:
      trust_headers: true
      # Optional: keep them only on connections from the proxy's network.
      trusted_peers: ["10.0.0.0/8"]
```

- `trust_headers` keeps the `x-authz-*` headers and the names
  `global.services.authz.identity` sets.
- `trusted_peers` (CIDRs) keeps them only when the connection's peer address is
  in one of the networks; the Router never reads `X-Forwarded-For` for this.
  Empty, every peer of a trusting listener is trusted.
- Each listener decides for its own requests, and a reload applies a change.
  Requests on a listener that doesn't trust identity are anonymous.

Memory, router replay and the per-user selection algorithms (`gmtrouter`,
`rl_driven`) record the user of each request. Without a trusting listener they
still load and treat every request as anonymous, and the Router logs one
warning at startup that names them.

### What needs `extproc`

Policy that enforces access by a client identity needs an identity source. The
Router refuses it at startup and on reload when no listener sets
`identity.trust_headers`, and the error names that option and
`--gateway extproc`:

- a decision on an `authz` (role binding) signal;
- a rate limit rule that matches `user` or `group`;
- `global.services.authz.providers`, which resolve per-user API keys.

Turn on `identity.trust_headers` on a listener behind an authenticating proxy,
or serve them with `--gateway extproc` behind a gateway that authenticates
clients. Token-bucket rate limiting, mTLS, JWT or OIDC, and advanced route
matching are planned for standalone mode; until then they need Envoy as well.

## Targets and platforms

`--platform cpu|amd|nvidia` works on both targets. It selects the image:
`vllm-sr` for the CPU, `vllm-sr-rocm` for AMD and `vllm-sr-cuda` for NVIDIA.

- **docker:** `amd` passes the ROCm devices through, and `nvidia` the NVIDIA GPUs
  (`--gpus all`).
- **kubernetes:** the generated Helm values set `gateway.mode`, the image
  repository for the platform, and a request for one GPU (`amd.com/gpu: 1` or
  `nvidia.com/gpu: 1`). `--target k8s` is the old name of `--target kubernetes`
  and works for this release only.

Without the CLI, the Helm chart takes the same value, `gateway.mode` (default
`standalone`), and the Operator runs the Router standalone unless
`spec.gateway` names a Gateway, which selects extproc. To move a release that
an Envoy-based gateway calls, see
[Upgrade and Rollback](upgrade-rollback#2a-helm-chart-upgrade).

### macOS

On macOS the docker target is CPU only: the built-in models run on the CPU in
the arm64 image, because Apple's virtualization gives Docker's Linux VM no Metal
or GPU compute. `--platform amd` and `--platform nvidia` fail there with a clear
message. GPU support through the host is tracked in
[#4636](https://github.com/vllm-project/semantic-router/issues/4636).

- **Comfortable on the CPU:** the 307M Vela task models (domain, PII, jailbreak
  and safety guards, embeddings, reranking), Vela Omni Nano, and the
  Decision 2.0 Kai-0.6B and Eos-0.8B decision models.
- **Larger models:** a model needs about 4 bytes per parameter on the CPU, so a
  2B decision model needs about 8 GB. Raise the memory of Docker's VM (Docker
  Desktop: Settings, Resources) above what the config's models need, or pick a
  smaller model. See [Choose a model](../model-runtime/choose-a-model).

## Options of `vllm-sr serve`

`vllm-sr serve --help` lists the options in groups. Each group applies to some
of the three ways `serve` runs: the Router on the docker target, the Router on
the kubernetes target, and engine mode (`vllm-sr serve MODEL`, the model
runtime in a container). An option used where its group does not apply is an
error that says where it applies.

| Group | Applies to | Options |
| --- | --- | --- |
| Common options | docker, kubernetes, engine mode | `--platform`, `--image`, `--log-level` |
| Router options | docker, kubernetes | `--config`, `--target`, `--gateway`, `--minimal`, `--readonly`, `--algorithm` |
| Container options | docker, engine mode | `--image-pull-policy`, `--container-runtime` |
| Docker target | docker | `--router-image`, `--envoy-image` (with `--gateway extproc`), `--dashboard-image`, `--startup-timeout`, `--replace-active-config`, `--recipe-env` |
| Kubernetes target | kubernetes | `--namespace`, `--context`, `--profile`, `--chart-dir` |
| Engine mode | engine mode | `--models`, `--revision`, `--device`, `--host`, `--port`, `--runtime-profile` |

`--container-runtime` (`docker` or `podman`) replaces `--runtime`, which works
for this release only, on `serve`, `status`, `logs`, `stop` and `dashboard`.
