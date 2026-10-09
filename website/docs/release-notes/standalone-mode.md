# Release note: standalone mode is the default

The first release that includes
[#4623](https://github.com/vllm-project/semantic-router/issues/4623) serves the
OpenAI-compatible API from the Router itself. `vllm-sr serve` starts no Envoy
container by default: the Router container publishes the config's `listeners`,
answers `/health` and `/ready` on them, and proxies to the model backends. The
routing decisions, the upstream requests and the responses are the same as
behind Envoy. See [Gateway Modes](../installation/gateway-modes).

## Keep the previous stack

```bash
vllm-sr serve --gateway extproc
```

`--gateway extproc` starts the Envoy container in front of the Router exactly
as earlier releases did, with the same Envoy configuration. Choose it for Envoy
features the standalone Router does not have yet, such as token-bucket rate
limiting, mTLS, JWT or OIDC, or advanced route matching. The published [comparison](../proposals/standalone-mode#results) measured a
specific synthetic workload and CPU allocation. It does not establish a
throughput advantage for every model, plugin or concurrency level; validate
the gateway choice with your own workload.

## What changes for a standalone stack

- **No Envoy container.** `vllm-sr status` and `vllm-sr logs` report the Router
  and the Dashboard; `vllm-sr logs envoy` and `--envoy-image` need
  `--gateway extproc`. The Dashboard's Playground and readiness checks use the
  Router's listener.
- **Proxy-control headers from clients are dropped.** `x-envoy-internal` and
  Envoy's retry, timeout and tracing controls (such as `x-envoy-max-retries` and
  `x-envoy-upstream-rq-timeout-ms`) never reach a backend, as behind Envoy.
- **Identity headers from clients are dropped,** both `x-authz-*` and the names
  `global.services.authz.identity` sets, because no authenticator stands in front
  of the Router. If a proxy or your application sets them, turn on
  `listeners[].identity.trust_headers` (optionally with `trusted_peers`).
  Without a trusting listener, a config with a decision on an `authz` signal, a
  rate limit rule that matches `user` or `group`, or authz providers is refused
  at startup, and memory, router replay and the per-user selection algorithms
  treat every request as anonymous.
- **Listener changes need a restart.** A reload that changes a listener's address,
  port, timeout or `tls` paths, or adds or removes a listener, is rejected as
  `restart_required`; run `vllm-sr serve` again.

## New

- `listeners[].tls` (`cert_file`, `key_file`) serves a listener over TLS in
  standalone mode, and a renewed certificate in those files serves new
  connections without a restart. `--gateway extproc` refuses it rather than
  serve the listener in cleartext.
- `listeners[].models` restricts a standalone listener to the request models it
  lists, for example a public key to `vllm-sr/auto` only; other models get
  `403 model_not_allowed` and `/v1/models` lists only the allowed names.
  `--gateway extproc` refuses it, since its Envoy listener does not enforce it.
- `--gateway standalone|extproc` and `--platform auto|cpu|rocm|cuda` work on the
  kubernetes target too. The CLI writes them into the generated Helm values as
  `gateway.mode`, the image repository and a GPU request.
- `vllm-sr serve --help` lists its options by group, and an option of another
  group is an error that says where it applies.
- The Router binary takes `-gateway standalone`. Its default stays `extproc`, so
  a manifest that runs it without the flag behaves as before.
- On macOS, `--platform rocm|cuda` fails with a clear message: Docker's Linux VM
  gets no GPU there, so the docker target runs the CPU image.
- **Timeouts, retries and fallback, per model and per decision.**
  `providers.models[].reliability` gains connect, total, idle, per-try and
  first-byte timeouts, retriable status codes, back-off, `Retry-After` and
  retry budgets. A decision's `reliability` and `fallback` blocks override them
  for the requests it routes, in both gateway modes; `first_byte_timeout` is
  standalone only. See
  [Tune timeouts, retries, and endpoint health](https://vllm-sr.ai/docs/installation/model-configuration#tune-timeouts-retries-and-endpoint-health)
  and [Fall back to another model](https://vllm-sr.ai/docs/installation/model-configuration#fall-back-to-another-model).
- **Configuration versions and rollback.** Every accepted change activates a
  numbered version, and each response names it in `x-vsr-config-version`. A
  rejected change leaves the active version serving and reports why. The last
  ten activations survive a restart, and `POST /api/v1/config/rollback`
  activates a recorded version as a new one. See
  [Configuration Management](../installation/configuration-management).

## Kubernetes runs standalone too

- **The Helm chart** sets `gateway.mode: standalone` by default. The Router
  serves the listeners in its config, the Service exposes their ports, and the
  probes check `/ready` and `/health` on them. The default listener is
  `http-8899`. `gateway.tls.secretName` mounts a TLS Secret as a volume, so a
  rotated certificate serves new connections without a restart.
- **The Operator** runs the Router standalone when `spec.gateway` is unset: the
  Router serves port 8801 itself, the port the Operator's Envoy sidecar served
  before. The sidecar is gone, and the Operator deletes its ConfigMap once the
  rollout completes. `spec.gateway` still selects the Gateway integration, with
  the Router serving ext_proc.
- **Keep a gateway in front.** If Envoy Gateway, Agent Router, Istio, KServe,
  llm-d or another Envoy-based gateway calls the Router over ext_proc, upgrade
  with `--set gateway.mode=extproc`; the integration values files under
  `deploy/kubernetes/` set it. An upgrade whose live config still has the old
  default listeners (`grpc-50051`, `http-8080`) fails to render in standalone
  mode, before anything changes. `helm rollback` restores the previous release.

## One router image

`vllm-sr`, `vllm-sr-rocm` and `vllm-sr-cuda` serve `vllm-sr serve`, the Helm
chart and the Operator, in either gateway mode, so Kubernetes gains CUDA. With
no arguments, or with Router flags, the image runs the Router on
`/app/config/config.yaml`, as the former `extproc` image did.

## The Dashboard needs no container socket

`vllm-sr serve` in an empty directory still opens setup in the Dashboard, and
now keeps waiting: when you activate a config there, the CLI starts the Router
(and Envoy with `--gateway extproc`) from it. If you stop the command first, the
next `vllm-sr serve` starts the Router, and `vllm-sr status` says that setup is
complete. The Dashboard no longer starts, stops or execs containers, and no
container socket is mounted into it.

A change the running Router hot-reloads applies as before. A change the running
containers can't take is saved and waits for the CLI: one the Router answers
`restart_required` for (in standalone mode, a listener's set, address, port,
timeout or TLS certificate), or, with `--gateway extproc`, one that changes
Envoy's generated configuration. The Dashboard answers "Restart required: run
`vllm-sr serve` to apply.", `vllm-sr status` reports it, and the next
`vllm-sr serve` recreates the containers from the saved config.

Recipe activation follows the same rule. A Recipe that changes the listeners,
the managed storage or the Router management API used to make the Dashboard
recreate the containers itself. Now the activation, once confirmed, is
committed and answered `202` with `status: restart_required` and that message,
and the next `vllm-sr serve` creates the containers from the Recipe's config,
starting any storage it adds. Storage the Recipe stops using keeps its data and
runs until `vllm-sr stop`. Deactivation works the same way.

The Dashboard image no longer contains a container CLI. Its status page reads
the Router's and Envoy's HTTP probes and the files `vllm-sr serve` keeps beside
the runtime config, so a Router waiting for first-run setup reads "standby",
one `vllm-sr serve` is starting reads "starting", and a stopped one reads "not
running", with what to run. Logs come from the bounded log spool, as before.

A Recipe with bearer authentication for the Router management API no longer
needs a root `vllm-sr serve`:

- `vllm-sr serve` owns the management credential the Dashboard uses: your
  `VLLM_SR_DASHBOARD_RECIPE_TOKEN`, or one the stack generates and keeps in its
  owner-only state. It passes the credential by name to the Dashboard, and to
  the Router whenever its config binds it. The Dashboard no longer keeps a copy
  in its Recipe store, and removes the one an earlier release left there.
- The Recipe store is shared with the group of the user who runs
  `vllm-sr serve`, so a non-root CLI reads the active package and recovers an
  interrupted activation. A store an earlier Dashboard wrote needs sharing
  once; `vllm-sr serve` prints the command. See
  [Security Hardening](../installation/security-hardening#recipe-store-permissions).

## Engine mode uses the same instance frontend

`vllm-sr serve ARTIFACT --engine` runs the frontend, Dashboard and
managed model pool using the selected platform image. It sets
`global.router.enabled: false`; starting again without `--engine` enables the
saved routing configuration. Mode is selected at startup, not in Dashboard.

- Native System One and decision requests use explicit listener model grants
  and the listener's API keys. Chat model permissions stay separate.
- Configure listener addresses and ports, multiple logical deployments and
  replicas in the canonical `--config` document.
- Positional MODEL, `--revision`, `--runtime-profile`, `-dp` and `--device-ids`
  configure the default judgment deployment without changing listener grants.
- Worker-level APIs such as classify, embeddings and rerank remain on
  `vllm-srun`; they are not automatically public frontend endpoints.

## Looper calls its models from the Router

A Looper algorithm, a prompt helper and context recovery now make their model
calls inside the Router, in either gateway mode, instead of sending them back
through the gateway. Each call runs the decision's plugins and goes to the
model's `backend_refs` with the provider model's timeouts and retries. The
responses are the same, with one exception below.

- **A failed model reads shorter.** Where a Fusion or Router Flow response lists
  a model that failed, its `error` is `answered 503` (with the status),
  `timed out`, `cancelled`, `invalid response` or `failed`, without transport
  details.
- **A model the Router calls needs `backend_refs`.** A decision whose Looper
  algorithm, prompt helper or context recovery calls a model without
  `providers.models[].backend_refs` fails to load, and the error names the
  decision and the model. `vllm-sr config validate` reports the same error. If an
  external gateway owns the backends (`listeners: []`), point that model's
  `backend_refs` at the gateway's OpenAI-compatible address.
- **`global.integrations.looper.endpoint` is deprecated.** The Router ignores
  it and logs `looper_endpoint_deprecated`, and `vllm-sr config migrate`
  removes it. The next release drops the field.

## Envoy mode changes too

These apply with `--gateway extproc` and behind a gateway integration:

- **Retries.** With `retry_count` set, a retry prefers an endpoint it has not
  tried yet (Envoy's `previous_hosts`), as in standalone mode. `retry_count`
  without `retry_on` takes the default retry conditions instead of failing
  validation.
- **Per-decision overrides reach Envoy.** The Router sends a decision's
  timeouts and retries as Envoy's per-request headers
  (`x-envoy-upstream-rq-timeout-ms`, `x-envoy-upstream-rq-per-try-timeout-ms`,
  `x-envoy-max-retries`, `x-envoy-retry-on`, `x-envoy-retriable-status-codes`).
  Every Envoy configuration the CLI renders or the repository ships lets
  ext_proc set exactly those five headers (`mutation_rules`); a custom Envoy
  configuration needs the same rule for the overrides to apply.
- **`header_mutation` cannot set those five headers.** Loading the config drops
  such entries with a warning. Envoy ignored them before, so no working
  configuration changes.
- **Duplicate listener names are rejected,** as Envoy already did.
- **A fallback answer carries the decision headers.** A response served by
  cross-model fallback carries `x-vsr-selected-decision`,
  `x-vsr-selected-algorithm`, `x-vsr-selected-recipe` and
  `x-vsr-routing-latency-ms` like any routed response, as in standalone mode.
  It used to name only the serving model (`x-vsr-selected-model`) and the
  attempt count (`x-vsr-fallback-attempts`).

## Removed

- **Fleet Simulator (`vllm-sr-sim`):** its package, its image, its PyPI and
  release jobs, its make targets and its documentation. Installed copies keep
  working; the v0.3 documentation still describes them.
  [Upgrade and Rollback](../installation/upgrade-rollback) shows how to remove
  the sidecar container an earlier `vllm-sr serve` started.
- **OpenClaw:** the agent integration is gone from the CLI, the Dashboard and
  the Helm chart's documented variables. That covers the Dashboard's OpenClaw
  page, the Playground's HireClaw mode and ClawRoom, the `/api/openclaw/*`
  endpoints, the `openclaw.read` and `openclaw.manage` permissions, and
  `vllm-sr config import --from openclaw`. For this release the Dashboard
  still accepts its `-openclaw*` flags and ignores them, like the
  `OPENCLAW_*` variables, and logs one `DEPRECATED` line naming those it was
  given. Remove them from custom manifests: the next release no longer
  accepts the flags, and an unknown flag stops the Dashboard at startup.
  `vllm-sr serve` no longer mounts the container runtime socket into the
  Dashboard.

## Renamed

| Before | Now | The old name |
| --- | --- | --- |
| `--target k8s` | `--target kubernetes` | works for this release, with a warning |
| `--runtime docker\|podman` | `--container-runtime docker\|podman` | works for this release, with a warning |
| image `extproc` | image `vllm-sr` | published with the same digests for this release |
| image `extproc-rocm` | image `vllm-sr-rocm` | published with the same digests for this release |
| `make docker-build-extproc` | `make docker-build-vllm-sr` (`-rocm`, `-cuda`) | removed |
