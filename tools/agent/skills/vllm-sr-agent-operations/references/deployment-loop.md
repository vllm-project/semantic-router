# Deployment and model-pool details

Follow the [operations skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md) for installation and capability
preflight. Check the installed commands and selected image capabilities before
runtime mutations; record concrete incompatibilities when a prerequisite fails.

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
lifecycle commands, preserve runtime/platform/image choices, and supply explicit
management/inference origins for remote checks. Management commands do not infer
a custom management port from YAML.

For a trial, prefer loopback publication and the minimal supported mode. Inspect
both canonical listener addresses and actual container port bindings. Split
containers may need an internal Router address of `0.0.0.0` even when its host
management port is restricted to loopback. Do not expose a private management
API just to make a remote browser convenient; use a tunnel when appropriate.

For an initial launch, validate locally and serve before asking for live plans.
Once the Router responds to `GET /api/v1`, follow its advertised readiness,
startup, and inventory operations. Process startup alone is not readiness.
Confirm backend reachability from the deployment network before routed probes.

## Accelerator and model preflight

Reserve Router model resources separately from generation backends. Estimate
resident weights from the selected precision, not a MoE model's active parameter
count. Include KV/runtime memory at the requested context and concurrency, and
storage for weights, images, and compilation caches. Verify device visibility
and driver/container compatibility before downloading the full pool.

When acceleration is required, verify the effective model bindings, successful
initialization, and actual devices in the live inventory. A runtime platform
selection alone does not establish each model's device.

For a local GPU deployment, explicitly select the `--platform` value supported
by the installed `serve --help` and detected hardware. The CLI does not select
a GPU platform automatically. Check inherited platform, image, and runtime
overrides before reusing a stack. Verify device passthrough, driver compatibility,
and the selected image. For Kubernetes, configure images and devices through the
deployment profile rather than the local `--platform` shortcut.
Keep Router devices separate from generation backends where required and retain
the same selection on restart. If the available runtime cannot meet the requested
acceleration, report that limitation before starting a different deployment.

If initialization or inference fails, retain the image, model revision, error,
and triggering input size. Qualify a proposed fix on that path; a successful
short request does not establish long-context or concurrent behavior. Use the
[evaluation checks](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) required by the user's task.

## Add or replace physical models

1. Verify the backend's protocol, credential reference, and supported request
   types. Query its model list and make a minimal direct request. Record both
   requested and returned model identity; aliases need not match. Verify the
   deployed variant's modalities, tool support, context, and output budget before
   advertising them in its Model Card. Match decision bindings to the backend's
   supported reasoning effort.
2. Add or update the provider, matching Model Card, and decision model reference.
   A provider or card alone does not make a model routable. Preserve the chosen
   Recipe's lane structure and minimum candidate counts.
   Check required quality indices and their effort/capability evidence. Missing
   evidence can exclude every candidate even with healthy, reachable backends;
   configuration validation and readiness do not establish branch eligibility.
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

Record limits at each layer: the learned deployment's `input.max_tokens` and
`overflow`, the decision's `rules.on_unknown`, Model Card context/output limits
used by `candidate_requirements.context: known_limits`, and the backend's total
prompt-plus-generation context. A learned binding's `overflow: reject` does not
by itself make that limit an API rejection boundary: an unknown signal can cause
`no_match` and another decision to serve the request. Candidate metadata can also
exclude a budget that the backend accepts directly. Verify these separately with
the [boundary checks](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md#token-boundaries-and-public-errors)
before advertising supported limits.

## Dashboard access

When UI access is requested, launch the Dashboard component with the intended
stack; `--minimal` excludes it. Local Docker publishes the Dashboard port on all
host interfaces by default. Inspect actual port bindings and establish the
intended network access controls before launch; an SSH tunnel alone does not
close a publicly published port. Preserve authentication state across lifecycle
operations and verify a real login. Dashboard sessions and inference credentials
are separate; do not hand off a temporary login token as a persistent API key.

The local CLI supports initial admin provisioning through `DASHBOARD_ADMIN_EMAIL`
and `DASHBOARD_ADMIN_PASSWORD`, with optional `DASHBOARD_ADMIN_NAME`. Supply secret
values through the environment, never command arguments or committed files.
Provision the account before access is opened and set
`DASHBOARD_ALLOW_OPEN_BOOTSTRAP=false` when registration is not intended; without
admin credentials the local CLI can enable open first-user registration.
Existing users are not reset by bootstrap variables. Inspect the installed
authentication contract when reusing an account or using the registration flow;
do not assume default credentials. Record the access URL/tunnel and hand off
credentials privately through the user's chosen secure mechanism.

Follow the main skill's UI path and the
[repeated checks](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md#repeated-api-and-ui-checks) to verify real
Playground completions. Server-backed preview and inference delivery are
separate evidence. Keep credentials and raw private workload outputs out of
source control and public receipts.

## Observability retention and checks

Local stacks provision Prometheus, Grafana and Jaeger unless `--minimal` is used.
Open the provisioned **vLLM Semantic Router Dashboard**, then check Router and
collector scrape health before interpreting an empty graph. A missing series
is not a successful zero, and an exporter success counter is not proof that all
requests were sampled or retained. Preserve the configured sampling policy.

Local Jaeger uses a stack-specific `<jaeger-container-name>-data` named volume
at `/tmp`, non-root Badger storage, and seven-day trace retention. Grafana uses
`<grafana-container-name>-data` at `/var/lib/grafana` for its database and
preferences, with the image's non-root user. Prometheus keeps its local TSDB with
15-day retention. Container replacement preserves these stores. An explicitly
authorized telemetry reset must remain separate from benchmark, authentication,
learning and configuration stores. Switching an older in-memory Jaeger instance
to Badger does not migrate its memory; a limited search is not a complete count.

For an authorized observability update, verify the same known trace across a
collector replacement using an isolated synthetic OTLP fixture without model
generation. Then inspect request stage parentage and duration when an inference
probe is separately authorized. The `semantic_router.request` root encloses
signals, decision, algorithm, plugins and upstream response. Bounded traffic
kinds separate public inference, authenticated internal attempts and catalog or
health polling. The final client status remains distinct from the upstream
status when a local guard blocks a response. Actual per-signal evidence preserves
missing values; projection scores and backend resolution are trace events.
Historical traces retain their original timing and attributes.

Use `llm_request_outcomes_total{traffic_kind="inference"}` for public request
outcomes, not model selections or error-event counters. Model first-response
observation and response-duration-per-output-token metrics describe the available
measurements, not first-token or decode-only latency. Missing measurements and
unsupported looper attempt accounting remain unknown. Recipe selection counters
describe selections, not completed model calls; consult usage separately.
