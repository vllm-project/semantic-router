# Deployment and model-pool details

Follow the [operations skill](../SKILL.md) for installation and capability
preflight. A version described as dev can still be stale; missing required
commands are a compatibility failure before any runtime mutation.

## Isolate the intended deployment

Inspect existing processes, containers, state directories, ports, and runtime
images before selecting a stack. Discover host acceleration vendor-neutrally.
Use installed `serve --help` for platform, image, Dashboard, and mode options;
record the exact package version and image identity actually launched.

Local Docker uses `VLLM_SR_STACK_NAME` and `VLLM_SR_PORT_OFFSET`. The offset
changes inference publication as well as management ports. A separate config
directory keeps its `.vllm-sr` state separate only when `VLLM_SR_STATE_ROOT_DIR`
is unset. Inspect that override and explicitly select a separate state root when
needed; an inherited root can otherwise load another stack's active config.
Check every effective port for collisions, keep the same environment for
lifecycle commands, and supply explicit management/inference origins for remote
checks. Management commands do not infer a custom management port from YAML.

For a trial, prefer loopback publication and the minimal supported mode. Inspect
both canonical listener addresses and actual container port bindings. Split
containers may need an internal Router address of `0.0.0.0` even when its host
management port is restricted to loopback. Do not expose a private management
API just to make a remote browser convenient; use a tunnel when appropriate.

For an initial launch, validate locally and serve before asking for live plans.
Once the Router responds to `GET /api/v1`, follow its advertised readiness,
startup, and inventory operations. Process startup alone is not readiness.
Confirm backend reachability from the deployment network before routed probes.

## Add or replace physical models

1. Verify the backend's protocol, credential reference, and supported request
   types. Query its model list and make a minimal direct request. Record both
   requested and returned model identity; aliases need not match.
2. Add or update the provider, matching Model Card, and decision model reference.
   A provider or card alone does not make a model routable. Preserve the chosen
   Recipe's lane structure and minimum candidate counts.
3. Validate and, for an existing Router, plan against its exact management origin.
   Changes to backend topology can require an Envoy restart. Activate through the
   deployment workflow within the user's existing authorization; obtain only
   missing authorization for disruption beyond that scope.
4. Confirm readiness and active config, preview affected branches, and probe
   actual delivery through each affected entrypoint. Assert selected-model and
   upstream response identity separately where the backend supports it.
5. Measure latency, cost, throughput, or task quality when those are part of the
   requested objective. A full benchmark is not an installation prerequisite.

Document capability adaptations explicitly. A text-only derivative must report
its unsupported image path and excluded decision; it is not a successful run of
the full multimodal baseline. Do not reduce candidate minimums or invent a model
to hide unavailable capability.

If Dashboard or Playground verification is requested, follow the optional UI
path in the main skill and verify a real streamed completion. Server-backed
preview and inference delivery are separate evidence. Keep credentials and raw
private workload outputs out of source control and public receipts.
