# Router Flow

## Overview

`workflows` runs a bounded, multi-step Router Flow behind one
OpenAI-compatible model name.

Flow coordinates model workers within that configured workflow. The calling
agent harness owns the outer agent loop, task state, and tool execution
permissions. When Flow returns a tool call, the harness executes it and sends
the result back so Flow can resume its pending workflow.

Expose `flow` through an ordinary `entrypoints` mapping to a recipe. The public name has no built-in dispatch behavior: the selected recipe evaluates its signals and decisions, and `algorithm.type=workflows` activates the algorithm. Use a dedicated recipe when this entrypoint should run only flow policies.

## Key Advantages

- Exposes a bounded multi-model workflow as one model name: `vllm-sr/flow`.
- Keeps worker boundaries explicit: dynamic planners may only use the decision's
  `modelRefs`.
- Supports both static role plans and dynamic planner-generated workflows.
- Records a Flow trace with plan, worker steps, responses, failed models, and
  usage.

## What Problem Does It Solve?

Some requests need orchestration rather than a one-step route decision: split the
task, ask multiple workers for targeted work, verify or reconcile the outputs,
and return one final answer through the same chat completions API. `workflows`
makes that orchestration part of Router policy while keeping the public model
surface as small as `vllm-sr/flow`.

## When to Use

- A route should expose a single model name but run a bounded multi-model workflow.
- The worker pool should come from the decision's `modelRefs`.
- You want static low-latency templates for predictable tasks.
- You want dynamic planner-generated workflows for harder reasoning, coding, or
  verification tasks.

## Configuration

Map the public name to the recipe shown below. Declare it as its own recipe, and point the entrypoint at it, to isolate it from the other recipes.

```yaml
entrypoints:
  - model_names: [vllm-sr/flow]
    recipe: default
global:
  integrations:
    looper:
      max_response_bytes_mb: 32
      flow:
        state:
          store_backend: file
          ttl_seconds: 1800
          file:
            directory: .vllm-sr/flow-state
```

Configure a dynamic Flow decision:

```yaml
recipes:
  - name: default
    decisions:
      - name: coding_flow
        description: Coordinate coding work through planned worker steps.
        priority: 100
        output_contract: Preserve any explicit output format exactly.
        modelRefs:
          - model: openrouter/gemini-pro
          - model: openrouter/deepseek
          - model: qwen/qwen3.6-rocm
        algorithm:
          type: workflows
          on_error: skip
          workflows:
            mode: dynamic
            planner:
              model: qwen-coordinator
              max_completion_tokens: 2048
            max_steps: 6
            max_parallel: 3
            round_timeout_seconds: 90
            min_successful_responses: 2
```

| Field | Use it for |
| --- | --- |
| `output_contract` | Decision-scoped prompt text: benchmark or application format requirements that apply across static Flow, dynamic Flow, Fusion and ReMoM, instead of task-specific prompts hard-coded in an algorithm. |
| `output_contract_spec` | Typed normalization and post-processing the Router executes, such as choice extraction or terminal-action JSON normalization. Only Fusion, ReMoM and Flow decisions accept it, and Flow refuses `reference_selection`, since its final lists worker outputs without numbers. |

Extraction defaults to exact `content` matching. Use `extract.sources` or
`extract.mode: json_object` only when the decision explicitly permits a wider
parser.

The planner model generates the control plan:

- **Default planner.** Omit `planner.model` to use the first assigned worker,
  in declared order, that is eligible for the complete planner request, JSON
  output and output/context budget included. This scan makes no model calls.
- **Explicit planner.** An override keeps that target and must pass the same
  stage checks; it is never replaced by another model on failure. It can be a
  helper outside the worker `modelRefs`, but needs a backend in
  `providers.models[].backend_refs`, or the configuration fails to load.
- **No eligible planner** fails the request closed.
- **Workers stay in `modelRefs`.** The executor rejects a plan that names a
  worker outside that list, and planner selection doesn't reduce a configured
  minimum of distinct successful workers.

Set `final.model` to choose a worker from `modelRefs` for the final answer,
independently of the planner. It works in both modes: configured `final.model`
and `final.prompt` override the corresponding fields of a generated plan.

A fast planner can then organize the work while a stronger model synthesizes
the answer. Check that the planner reliably returns valid JSON before relying
on this split.

Reasoning controls come from each model's
[reasoning configuration](../../../installation/model-reasoning.md) and decision
reference. For a custom model, declare the reasoning family before using
`use_reasoning: false`; without a family, the backend's reasoning behavior
passes through.

:::warning Budget the whole run
Planning, sequential worker steps and final synthesis must fit within the
client and gateway timeouts; a per-round timeout does not extend the gateway
deadline. Check the complete output and `finish_reason`, not only HTTP success.
:::

Static mode uses an explicit role plan. Each role model must be in the
decision's `modelRefs`.

```yaml
recipes:
  - name: default
    decisions:
      - name: static_flow
        description: Coordinate a fixed sequence of worker roles.
        priority: 100
        modelRefs:
          - model: qwen-worker
          - model: deepseek-worker
        algorithm:
          type: workflows
          on_error: skip
          workflows:
            mode: static
            roles:
              - name: thinker
                models: [qwen-worker]
              - name: worker
                models: [deepseek-worker]
              - name: verifier
                models: [qwen-worker]
            final:
              model: qwen-worker
            max_steps: 3
            max_parallel: 1
            round_timeout_seconds: 90
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state.store_backend` | string | `file` | Pending tool-call workflow state backend: `memory`, `file`, or `redis` |
| `state.ttl_seconds` | int | `1800` | TTL for pending tool-call workflow state |
| `mode` | string | `static` | `static` role execution or `dynamic` planner-generated execution |
| `template` | string | `micro_agent` | Static workflow template name |
| `roles` | list[object] | required for static | Ordered static roles, each with `name`, `models`, optional `prompt`, and optional `access_list` of earlier role ids or agent ids |
| `final.model` | string | plan's final model, then planner, then first worker response | Override the final synthesis model in either mode; must belong to `modelRefs` |
| `final.prompt` | string | plan's final prompt or built-in synthesis prompt | Override the final synthesis instruction in either mode |
| `planner.model` | string | first eligible assigned worker | Optional explicit model used to generate the workflow plan |
| `planner.max_completion_tokens` | int | `2048` | Max completion tokens for the planner JSON plan only |
| `minimum_candidates` | int | unset | Minimum distinct decision `modelRefs` required after Recipe materialization and context eligibility filtering |
| `max_steps` | int | `3` | Maximum workflow steps accepted from the planner |
| `max_parallel` | int | `2` | Maximum worker models per step |
| `max_completion_tokens` | int | request default | Max completion tokens for worker and final synthesis calls |
| `round_timeout_seconds` | int | unset | Maximum seconds to wait for each workflow step or final synthesis |
| `min_successful_responses` | int | all models | Continue a parallel step once this many workers succeed |
| `temperature` | float | request default | Temperature for planner, worker, and synthesis calls |
| `include_intermediate_responses` | bool | `true` | Include Flow plan and worker outputs in the response trace |
| `algorithm.on_error` | string | `fail` | Set beside `type`, not in this block: `fail` on worker error or `skip` failed workers when at least one worker succeeds |

Every static role and every planner-generated step must contain at least
`min_successful_responses` Models. A plan that cannot satisfy its configured
quorum is rejected instead of running with a silently reduced quorum.

## Tool And Function Calling

Router Flow preserves the normal OpenAI-compatible tool-calling contract for
clients. Send `tools` or legacy `functions` on the `vllm-sr/flow` request as you
would for a single model.

When a worker or the final synthesizer returns `tool_calls`, Flow:

1. stores the pending workflow state, including plan, completed step outputs,
   current agent request, and that agent's private tool trajectory;
2. rewrites each `tool_call_id` with a Flow state prefix and returns the tool
   call to the client;
3. consumes the state on the next request when the client sends matching
   trailing `tool` messages;
4. routes those tool results back to the exact worker or final agent that
   requested them, without replaying unrelated workers;
5. continues that agent's tool loop until it produces content, then resumes the
   remaining workflow.

Each worker has its own message history. A later step's `access_list`
exposes only prior step or agent outputs, never another worker's raw tool calls
or tool-result trajectory:

| `access_list` | Exposes |
| --- | --- |
| omitted | All earlier step outputs |
| `[]` | Nothing: the step is isolated from prior outputs |
| a role id, such as `solver` | All outputs of that role |
| an agent id, such as `solver:1:deepseek-worker` | One worker of a parallel role |

The same agent id appears as `flow.steps[].responses[].agent_id` when
`include_intermediate_responses` is enabled.

Step IDs:

- **Unique.** Step IDs must be unique and must not collide with generated
  agent IDs; generated default IDs and normalized static role IDs follow the
  same rule.
- **Trimmed.** The Router trims step IDs before validation, as it does
  access-list entries and agent IDs.
- **Checked before any worker runs.** An ambiguous plan is rejected. In dynamic
  mode, `on_error: skip` uses the existing fallback plan instead.

The same validation applies when a persisted workflow resumes:

- A paused plan with conflicting IDs, or whose stored step identity no longer
  matches, is rejected without further model calls.
- A failed resume validation never starts a fallback workflow, even with
  `on_error: skip`. Restart with an unambiguous plan instead.

For local single-process development, `memory` is enough. For local restarts use
`file`. For multi-replica deployments, use `redis` so a tool-result turn can be
claimed by whichever router instance receives it.

## Request

```json
{
  "model": "vllm-sr/flow",
  "messages": [{"role": "user", "content": "Debug this flaky test and propose a patch."}]
}
```

## Design Notes

Router Flow intentionally keeps the user-facing API small. The decision's
`modelRefs` are the worker pool. `algorithm.workflows` describes how to
orchestrate that pool, not a second model catalog.

Planner and worker models receive request-derived content according to the
workflow plan. Tool-call state can be persisted in memory, files, or Redis;
choose a backend, TTL, authentication, and encryption appropriate for that
content. See a complete example:
[`config/fragments/algorithm/looper/workflows.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/workflows.yaml).

## Optional trace transport

Router-generated `flow` evidence is preserved in OpenAI Chat Completions JSON
and SSE responses, including tool-call responses where the algorithm supports
them. The trace is optional: if adding it would exceed the complete response
limit or an individual SSE frame limit, the router omits the whole trace while
serving the valid answer. It does not truncate trace JSON or answer text.

OpenAI Responses and Anthropic Messages responses omit this Chat-specific
extension. When it is dropped, for an unsupported target protocol or for size,
`x-vsr-warnings` reports action `dropped`, field `flow`, and reason
`router_extension_unsupported_protocol` or `router_extension_size_limit`,
immediate Looper responses included.

Required answer content stays subject to the normal protocol limits; an answer
that cannot fit is rejected.

Provider fields do not acquire router provenance by using the same name. The
router validates provider content separately from its internal trace data.
