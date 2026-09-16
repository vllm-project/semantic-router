---
name: vllm-sr-agent-operations
description: Install, configure, verify, and improve vLLM Semantic Router through its CLI and Router API. Use for deployment and routing operations, with Dashboard verification when requested.
---

# vLLM Semantic Router

Follow the steps below with the CLI and Router API. Use the Dashboard when the
user requests it. Load linked details only for the current task; the user's
instructions and existing authorization take precedence over this skill.

## 1. Identify the target

Inspect the host, installed CLI, running stack, configuration, available
resources, and generation backends. Establish whether this is a new deployment
or an update to a particular stack. Preserve unrelated workloads and settings.

Use installed CLI help and `vllm-sr config schema` for supported commands and
fields. For a running Router, discover its operations through `GET /api/v1`
and its advertised schema/OpenAPI. Set `ROUTER_ORIGIN` to that stack's management
origin and pass `--endpoint` explicitly for online commands.

Read [deployment details](references/deployment-loop.md) when selecting a
runtime, accelerator, isolated stack, or remote access path. Size Router models
and generation backends separately for the requested context and concurrency.

## 2. Install when needed

For installation or an authorized upgrade, use the latest published dev package
unless the user selects another channel or version:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel dev --mode cli --runtime skip --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr --version
vllm-sr serve --help
vllm-sr config schema
```

For a custom install location, use the launcher's printed directory. Confirm the
commands needed for the task before starting services. Resolve a reported
incompatibility against the installed CLI and selected runtime; record their
versions rather than inferring compatibility from a channel name.

## 3. Configure and activate

For a new stack, start with `vllm-sr config init`. For an existing stack, use a
fresh `vllm-sr config get --endpoint "$ROUTER_ORIGIN"` as the candidate's baseline.
Inspect relevant schema sections rather than guessing fields. Replace starter
placeholders and keep credentials in environment variables.

Connect providers to the decisions that use them. Check backend reachability
and actual model capabilities, including context and output limits. When the
user selects a built-in Recipe, discover it with `vllm-sr recipe builtin list`
and read [configuration details](references/configuration-loop.md) for binding,
activation, and recovery. Preserve its required candidates and quality evidence;
report missing prerequisites rather than inventing scores or weakening the policy.

**New stack:** validate locally, then launch with the selected deployment options:

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

**Existing stack:** validate and plan against its exact management origin:

```bash
vllm-sr config validate --config candidate.yaml
vllm-sr config plan --config candidate.yaml --endpoint "$ROUTER_ORIGIN"
```

Apply a hot-reloadable change with `config apply`. For `RESTART_REQUIRED`, use
the authorized replacement flow; local Docker uses
`serve --config candidate.yaml --replace-active-config`. Ordinary `serve`
preserves active edits. Recipe packages have their own activation workflow;
see the configuration reference before replacing one.

After either path, confirm the Router's advertised readiness, active revision,
and published inference entrypoint. A Recipe name is not necessarily an
entrypoint; `vllm-sr/auto` does not automatically select a named Recipe.

## 4. Verify routing and delivery

Set `ENTRYPOINT` and `INFERENCE_BASE_URL` from the deployed entrypoint and actual
listener bindings. Check a representative routing decision and a real response:

```bash
vllm-sr route preview \
  --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --prompt 'Explain why this request should take this route.' --trace --json --timeout 300

vllm-sr route probe \
  --config config.yaml --base-url "$INFERENCE_BASE_URL" --model "$ENTRYPOINT" \
  --prompt 'Return exactly: route-ok' --timeout 300
```

Management and inference credentials are separate. Use the installed
`--token-env` and `--api-key-env` options with variable names, never secret values.

Preview runs Router signals and decisions; probe sends a request through Envoy
for backend generation. Inspect the selected Recipe, decision, backend, and
completed response. Add supported `--expect-*` assertions for the intended route;
calibrate the backend's returned model identity with a direct request first.
Choose completion and timeout budgets appropriate to the model.

Test the branches affected by the task and retain failed attempts. Read
[evaluation details](references/evaluation-loop.md) for tools, modalities,
token boundaries, repeated tests, or benchmarks. Encoder input limits and
backend prompt-plus-output context are separate budgets. Readiness or one
successful prompt does not prove all branches or sustained-load stability.

For requested Dashboard work, follow [access setup](references/deployment-loop.md#dashboard-access),
verify login, and complete a real Playground request using the same entrypoint.
A page load or simulated preview is not an inference test.

## 5. Improve and hand off

Capture the requested quality, cost, latency, resource, or reliability baseline.
Make one coherent change and repeat the same requests and workload. Keep it when
it meets the objective without violating hard constraints; otherwise recover
the previous configuration. Use [Recipe tuning](references/recipe-tuning.md)
for policy changes and the evaluation reference for requested benchmarks.

Leave the user with the config path, stack identity, access method, active
revision, package/image versions, verification results, and remaining limits.
Keep secrets and private request content out of public artifacts. Preserve the
user's existing authorization; ask only when an action extends beyond it.
