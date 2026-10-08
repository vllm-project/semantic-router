# Managed instance modes

The frontend stays running in both modes. `global.router.enabled` defaults to
`true`: Router mode adds the recipe routing pipeline and Chat/Responses access;
Engine mode sets it to `false` and serves native System One inference without
constructing recipe classifiers or upstream pools. Saved routing configuration,
listeners, Dashboard and model workers are retained. Mode and model placement
are independent: one logical deployment may use one worker or a replica pool.

```bash
vllm-sr serve vllm-sr/Vela-2.0-4B --engine --platform rocm -dp 2 --device-ids 0,1
vllm-sr instance --config config.yaml status
vllm-sr instance --config config.yaml models
# Restart with routing enabled, retaining model resources and routing policy.
vllm-sr serve --config config.yaml
```

`--engine` (`-e`) selects Engine mode at startup. Every invocation without it
starts Router mode, including when the saved configuration previously disabled
routing. MODEL replaces the configured default judgment deployment's artifact;
unspecified profile and replica placement remain unchanged. Listener grants
never widen. A new Engine configuration needs no Chat backends or routing YAML.
Health repair observes the running instance's active mode; it does not issue a
new user `serve` request or change that startup choice.

Both modes expose `POST /v1/systemone`, its native alias `POST /v1/decisions`, and
`GET /v1/systemone/models` on the standalone frontend. Dashboard can provide an
optional stable bridge but is not required for inference. These routes use the
listener's `api_keys` and separate, explicit `systemone.models` grant. Chat
`listener.models` does not grant System One access. Public IDs are deployment
`public_name` values or Hub artifact IDs; local artifacts require a public name.
The frontend uses its generation's retained model lease, independently of the
optional Router pipeline. A mode change cannot redirect a pinned native request
to a new deployment. Multiple listeners never union their publication scopes.

Discovery describes the explicit publication grant; `instance models` describes
actual readiness. Deploying or scaling a model never widens a grant. Under Engine
mode Chat, Responses and Chat model discovery are disabled. Worker-level classify,
embedding, rerank and bundle APIs remain worker APIs, not new public frontend
routes. Explicitly enabled global stores remain frontend-owned management
services; each holds only its own embedding consumer, independent of dormant
recipe signals. Minimal Engine configurations do not enable these stores.

The local Docker/Podman CLI attaches a host controller after startup with
Dashboard. `GET /api/instance` reports desired mode, observed active-snapshot
mode, active deployment, ownership and durable operation state.
`GET /api/instance/models` reports model inventory. These Dashboard endpoints
are read-only; there is no public mode deployment endpoint or `can_switch`
capability. Model deployment and scaling use the canonical model resource
configuration, independently of startup mode. `instance status` and `models`
are the public CLI inspection commands.

The host controller retains its private journal and bounded deployment
transactions for lifecycle recovery. Its Unix socket is an internal control
surface, not a browser mode-switch API.

A rejected candidate leaves the active generation serving. The controller
restores its previous canonical document when recovery is needed, using the
same API's compare-and-swap guard so another editor's changes are never
silently overwritten. Interrupted operations recover from the private journal
before accepting more work. Hash/generation equality is not an operation gate;
active mode, selected deployment and actual model readiness are verified.

Private state lives under `$XDG_STATE_HOME/vllm-sr/instances` (default
`~/.local/state/vllm-sr/instances`). Only the restricted Unix socket directory is
mounted read-only into Dashboard; no Docker socket, manifest or journal is
exposed there. Retain host state and the same stack/config environment across
controller restarts. A host supervisor can run:

```bash
vllm-sr instance --config config.yaml controller
```

Kubernetes and externally owned frontends remain owned by their deployment
system; Dashboard does not claim permission to control their host. Attached
external model workers retain their own owner even in a locally managed pool.

An image rollout using the canonical lower-level startup helper attaches the
controller after readiness with the actual runtime configuration and gateway:

```bash
vllm-sr instance --config config.yaml attach --runtime-config /path/to/active/runtime-config.yaml --gateway standalone
```

After changing the installed CLI package, restart an idle controller through its
host supervisor. No browser request supplies an image, command, host path or
upstream URL. External health repair should wait while an operation is active;
it can check `instance models` for native readiness in either mode.
