---
name: vllm-sr-agent-operations
description: Install, configure, verify, and improve vLLM Semantic Router through its CLI and Router API. Use for vLLM SR deployment and routing operations; include Dashboard verification when the user requests it.
---

# vLLM Semantic Router

Use the CLI and Router API directly. The Dashboard is optional. The user's
instructions, existing authorization, and deployment boundaries take precedence
over this skill.

## Work progressively

Follow the stages below: identify the target and prerequisites, configure and
activate, verify routing and delivery, then optimize against that verified
baseline. Load a linked reference only when its stage needs more detail. Keep
failed checks and affected branches visible while continuing independent work;
readiness or a passing subset does not establish a working complete Recipe.

## Authoritative contracts

- Treat installed CLI help, the running Router's discovery response, JSON
  Schema, and OpenAPI document as the current source of truth. Record package
  version and runtime image identity; a channel name alone does not establish
  compatibility.
- Discover progressively: start with `vllm-sr config schema`, then request only
  relevant `--section` or `--surface` views. Follow child paths; use `--expanded`
  for a self-contained section and `--full` only when needed.
- Discover Router operations from `GET /api/v1`; request the full or filtered
  `/openapi.json` when operation details are needed. Use the intended stack's
  management origin. Do not reuse remembered fields or endpoints when discovery
  is available.

## Install and select the target

1. Inspect the requested host, existing CLI, configs, containers, listening
   ports, model endpoints, runtime, and accelerator. Check accelerator inventory
   vendor-neutrally; a missing NVIDIA or AMD utility alone does not prove there
   is no GPU. Identify whether this is a new isolated deployment or a change to
   a particular running stack.
2. When installation or an authorized upgrade is needed, default to the latest
   **published dev package**, unless the user requests another channel or version.
   Install without starting a runtime:

   ```bash
   curl -fsSL https://vllm-sr.ai/install.sh | \
     bash -s -- --channel dev --mode cli --runtime skip --no-launch
   export PATH="$HOME/.local/bin:$PATH"
   vllm-sr --version
   ```

   For a custom install location, use the launcher's directory printed by the
   installer. Preserve an existing installation unless changing it is in scope.
3. Check capabilities before writing config or starting services:

   ```bash
   vllm-sr serve --help
   vllm-sr config schema
   vllm-sr config init --help
   vllm-sr config validate --help
   vllm-sr config plan --help
   vllm-sr route preview --help
   vllm-sr route probe --help
   ```

   Verify the commands and flags needed for the selected path, including
   `config apply` and `serve --replace-active-config` for existing-stack
   mutations. If missing, report the exact version and unsupported command.
   Resolve a compatible package or source build within the authorized scope
   before runtime changes; do not silently translate this workflow to obsolete
   commands.
4. Select the host-appropriate deployment path from installed help. For an
   additional local stack, discover and use its stack identity, state location,
   and port controls; inspect all resulting ports for collisions. Current local
   Docker uses `VLLM_SR_STACK_NAME` and `VLLM_SR_PORT_OFFSET`; the offset also
   affects inference listener host ports. Inspect `VLLM_SR_STATE_ROOT_DIR`, which
   overrides the config directory's state location, before selecting an isolated
   stack. Keep the same stack, state, runtime, platform, and image selections for
   lifecycle commands. Set `ROUTER_ORIGIN`
   to the selected management origin and pass `--endpoint` explicitly for online
   operations; CLI defaults do not infer a custom management port from YAML.
   Prefer the supported minimal mode when no UI or observability is requested.

   Keep trial listeners on host loopback unless exposure is requested. Inspect
   actual container port bindings after launch: an internal container bind
   address is different from host publication. Split Docker may require the
   Router to bind internally to `0.0.0.0` while publishing management on host
   loopback. Do not blindly set every internal address to `127.0.0.1`.
5. Size the Router's signal models and generation backends separately, including
   weights, runtime/KV memory, cache storage, and the intended context/concurrency.
   A platform flag selects runtime support; it does not prove every model binding
   uses a GPU. For required acceleration, inspect the selected Recipe's bindings
   and device settings, then verify actual devices in startup evidence and the
   live model inventory. See [deployment details](references/deployment-loop.md).

## Configure and start or update

1. Discover the relevant config sections before editing:

   ```bash
   vllm-sr config schema --section providers.models
   vllm-sr config schema --section routing.modelCards
   vllm-sr config schema --section routing.decisions.modelRefs
   ```

   Use `--surface KIND:NAME` for selected signals, projections, algorithms, and
   plugins. When a Router is already running, use its schema through the
   discovered endpoint option to check the deployed contract.
2. For the selected existing stack, read `vllm-sr config get --endpoint "$ROUTER_ORIGIN"`; for a new stack,
   use `vllm-sr config init`. When the user selects a built-in Recipe, first
   discover packaged resources without requiring a source checkout:

   ```bash
   vllm-sr recipe builtin list
   vllm-sr recipe builtin export --help
   vllm-sr recipe builtin init --help
   ```

   A bundle can contain several named recipes: `balance`, for example, belongs
   to `mom-v1`; it does not need a standalone directory. Use `builtin export`
   to inspect the verified bundle, or `builtin init` to select the named recipe
   and bind its decisions to providers from your config. Read the listed
   candidate requirements and supply explicit bindings for each decision that
   calls a backend. Check required quality evidence as well as candidate counts:
   a healthy backend can still be excluded when a required index is unavailable.
   Record the affected branch as blocked; do not invent scores or weaken the
   selector to make a probe pass. Immediate-response decisions, such as Vault's
   `guard`, need no model assignment. Preserve recipe structure and algorithm minimums; do not replace
   it with a similarly named source example or silently delete lanes to force
   validation. When the user authorizes a capability-specific derivative, use
   explicit adaptation options such as `builtin init --exclude-decision`.
   Preserve the original bundle as the baseline and record its identity, excluded
   decisions, and unsupported request types. Verify the remaining branches and
   describe the result as an adapted Recipe; it is not evidence that the full
   built-in Recipe ran successfully.
   Do not clone the repository merely to obtain already packaged resources.

   Replace starter placeholders and check listener
   addresses before launch. Preserve unrelated fields. Each routable physical
   model needs a provider entry and a decision model reference. Named recipes
   place decisions under `recipes[].routing.decisions`; built-in recipes receive
   model assignments when an Entrypoint is published. Keep optional model metadata
   in the shared top-level `routing.modelCards`, adding metadata required by the
   selected algorithm or capability. Keep credentials in environment variables
   and only environment references in config.

   Check backend reachability from the deployment network. Before asserting
   routed response identity, inspect the backend's model listing and make a
   minimal direct completion where available. Record the requested model and
   returned top-level `model`; backend aliases can differ from provider names.
   If direct verification is unavailable, report that limit and avoid inventing
   an expected response identity.
   Align Model Card metadata with the deployed backend's actual modalities,
   context, and output limits; align decision bindings with supported reasoning
   effort. Published model maxima are not proof of the local serving configuration.
   Report Router signal input limits separately; the backend's context window
   alone does not establish the routed input limit.
3. **New stack:** locally validate, then perform its initial launch using the
   selected deployment options:

   ```bash
   vllm-sr config validate --config config.yaml
   vllm-sr serve --config config.yaml
   ```

   Do not run `config plan` against an absent Router or an unrelated existing
   stack. Inspect service status, query the new Router's `GET /api/v1`, and use
   its advertised readiness operation. Wait for readiness before routing checks.
   Record the live config revision and refresh the relevant deployed schema.
4. **Existing stack:** locally validate, then plan the exact mutation against
   the selected Router. Derive the candidate from a fresh canonical `config get`,
   preserving active defaults and unrelated settings; see
   [configuration details](references/configuration-loop.md).

   ```bash
   vllm-sr config validate --config config.yaml
   vllm-sr config plan --config config.yaml --endpoint "$ROUTER_ORIGIN"
   ```

   Apply a hot-reloadable candidate within the requested scope with
   `vllm-sr config apply --config config.yaml --endpoint "$ROUTER_ORIGIN"`.
   For `RESTART_REQUIRED`, do not
   call the apply API. Once disruption of that stack is authorized, use the
   supported replacement flow; for local Docker:

   ```bash
   vllm-sr serve --config config.yaml --replace-active-config
   ```

   Without this flag, `serve` preserves active Dashboard or Recipe changes.
   The flag cannot replace an active Recipe package: discover and use its
   Recipe workflow. Recheck readiness and active revision after either path.
5. For a named Recipe, inspect `vllm-sr recipe --help` and the running Router's
   advertised Recipe and model/entrypoint operations. Discover its actual
   published entrypoint and use that in tests. Do not assume a Recipe name is
   an inference model ID. `vllm-sr/auto` is reserved; do not rebind it to a custom
   Recipe or assume it selects the named Recipe just activated.

## Verify routing and delivery

Set the following nonsecret values from the selected stack's discovered
entrypoint and actual host bindings, then check both paths:

```bash
vllm-sr route preview \
  --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --prompt 'Explain why this request should take this route.' --trace --json --timeout 300

vllm-sr route probe \
  --config config.yaml --base-url "$INFERENCE_BASE_URL" --model "$ENTRYPOINT" \
  --prompt 'Return exactly: route-ok' --timeout 300 \
  --expect-selected-model "$SELECTED_MODEL" \
  --expect-response-model "$RESPONSE_MODEL"
```

Set a request timeout appropriate for cold startup and long inputs; the examples
allow 300 seconds. A timeout is failed or incomplete evidence. Management commands
use `--token-env` (default `VSR_MGMT_TOKEN`); routed probe uses `--api-key-env`
(default `OPENAI_API_KEY`). Pass only environment variable names, never values.

Preview evaluates signals and decisions, including configured classifier and
embedding inference, without backend generation. Probe sends a real
request through Envoy. `--base-url` accepts the listener origin or its OpenAI
`/v1` root. Set `SELECTED_MODEL` from the expected routing decision and set
`RESPONSE_MODEL` from the direct backend calibration; omit the response assertion
when the backend does not expose a stable identity, and state the evidence limit.
Also assert the selected Recipe when testing Recipe binding and the installed
probe supports it. A selected-model header alone does not prove delivery.

Test representative branches for the requested change; one successful prompt
does not establish coverage of every route. For requested stability testing, use
the [repeated API and UI checks](references/evaluation-loop.md#repeated-api-and-ui-checks)
and report counts, duration, concurrency, failures, and tested limits. Keep config,
active revision, preview traces, probe results, and runtime identity together.

## Optional Dashboard verification

Use this path when the user requests Dashboard or Playground work. Discover the
installed Dashboard launch/access options and reuse the intended stack. Keep
remote UI access local or tunneled unless public exposure is authorized.
`--minimal` omits the Dashboard. When adding it, preserve the stack's runtime,
platform, images, and state, and complete the supported authentication/bootstrap
flow before testing. Verify login and session renewal; do not assume a
default account. A tunnel does not restrict an already public container port;
check publication and access controls before handing off the URL. See
[deployment details](references/deployment-loop.md#dashboard-access).

Verify the active Recipe and published entrypoint shown in the UI. If testing
Playground, send a real request and verify streamed backend output and completion;
opening the page is not an inference test. If testing routing preview in the UI,
use its server-backed preview and confirm a Router response. Label any simulated
preview as simulation and do not present it as live routing evidence. Report UI,
routing, and backend results separately when they differ.

## Optimize a verified baseline

Capture the requested quality, cost, latency, resource, or reliability baseline
first. Make one coherent change, keeping unrelated variables fixed, and repeat
the same cases and workload. Retain the candidate only when it meets the
objective without violating hard constraints.
Recheck capability metadata and affected branches after changing model precision,
GPU count, context, or concurrency. Recover the previous configuration when a
candidate regresses. A short feature smoke test does not qualify long-context or
sustained-load stability. Run model or Mixture-of-Models benchmarks when requested
or needed for the agreed objective; begin with
`vllm-sr benchmark intelligence plan --help`.

## Boundaries and handoff

- Keep secrets out of logs, command arguments, source control, and public
  artifacts. Deliver credentials privately when the user explicitly requests
  them, using the requested secure access method.
- Preserve unrelated workloads and user configuration. Resolve exact container
  and file targets; obtain missing authorization before destructive changes,
  public exposure, or disrupting another service. Do not ask again for actions
  already authorized within the requested deployment.
- Leave the user with the config path, stack identity, access method, active
  revision, package/image identity, validation and routing evidence, and remaining
  limitations. Do not equate preview with model quality or simulation with a
  live result.

Read [configuration details](references/configuration-loop.md) for compare-and-swap,
Recipe activation, or rollback; [deployment details](references/deployment-loop.md)
for isolation and model-pool changes; and
[evaluation details](references/evaluation-loop.md) for branch coverage or a
requested benchmark. Load only the reference needed for the current step.

Use [recipe tuning](references/recipe-tuning.md) to improve signals, projections,
decisions, retrieval, or agent continuity against representative requests.

See the [Router API](https://vllm-sr.ai/docs/api/router) and
[agent evaluation loop](https://vllm-sr.ai/docs/benchmarking/agent-evaluation-loop)
when the task needs the full contract.
