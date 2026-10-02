# Deployment and access

Use this reference when installing, changing runtime topology or opening UI
access. Discover supported flags with the installed `vllm-sr serve --help`;
choose platform and image from the actual host, not an assumed accelerator.

## Stack identity

Local stacks use `VLLM_SR_STACK_NAME` and `VLLM_SR_PORT_OFFSET`. The offset affects
inference and management ports. Inspect `VLLM_SR_STATE_ROOT_DIR`: an inherited
root can load another stack's active configuration even from a new directory.
Set this variable to the directory that contains `.vllm-sr`, usually the config
directory; pointing it at `.vllm-sr` creates a second, empty state directory.
Keep the selected state root, runtime, image, platform and ports consistent across
lifecycle commands. Supply the actual management endpoint explicitly.

Validate locally before a first launch. After startup, use Router discovery to
check readiness, active configuration and backend reachability from its network.
For existing deployments, use the [configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md).

Prefer loopback publication for a private trial. Inspect actual container port
bindings as well as listener YAML; split containers may need an internal Router
address of `0.0.0.0` while the host management port remains private. A tunnel does
not close an already public port.

## Hardware and physical models

Size Router models separately from generation backends. Include resident weights,
KV/runtime memory at the intended context and concurrency, and disk space.
The CLI does not select a GPU platform automatically; verify device visibility,
driver/image compatibility and the effective model bindings. Kubernetes devices
and images belong in the deployment profile.

When adding or replacing a backend:

1. Verify its protocol, requested/returned model identity and required capabilities
   with a bounded direct request. Keep credential values in environment variables.
2. Connect the provider, Model Card and recipe decision. A provider or card alone
   does not make a model routable. Preserve candidate-count requirements and use
   measured quality evidence; missing evidence can exclude healthy backends.
3. Plan and activate the change, then preview and probe the affected entrypoints.
   Backend/listener topology may require Envoy replacement.

Encoder input limits, candidate eligibility and backend prompt-plus-output context
are distinct. Check the relevant [boundaries](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md#token-boundaries-and-public-errors)
before advertising supported limits. A text-only derivative must disclose excluded
modalities rather than claim full recipe coverage.

## Dashboard access

`--minimal` omits Dashboard. Local Docker publishes its port on loopback by
default; set `VLLM_SR_DASHBOARD_HOST_BIND=0.0.0.0` only when external access is
intended.
Inspect actual port bindings before opening access. Preserve authentication state
when updating a stack and reuse an existing valid session where available.

Jaeger, Prometheus, and Grafana publish on loopback by default. Set
`VLLM_SR_OBSERVABILITY_HOST_BIND=0.0.0.0` only when their host ports need external
access; inspect the resulting bindings and secure access before exposing them.

Initial admin provisioning supports `DASHBOARD_ADMIN_EMAIL`,
`DASHBOARD_ADMIN_PASSWORD` and optional `DASHBOARD_ADMIN_NAME`. Supply secrets
through environment variables. Set `DASHBOARD_ALLOW_OPEN_BOOTSTRAP=false` when
registration is not intended; bootstrap variables do not reset existing users.
Dashboard sessions and inference API credentials are separate.

Verify the requested UI flow and real routed delivery through the published
entrypoint. Hand off the access URL/tunnel and credentials through the user's
chosen private mechanism. Runtime-specific storage and observability settings
are discoverable from the deployed config and product documentation.
