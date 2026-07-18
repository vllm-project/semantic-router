# Router Flow

## Overview

`workflows` is a **looper** algorithm for Router Flow: a single model name can
run a bounded micro-agent workflow behind the OpenAI-compatible API.

It aligns to `config/algorithm/looper/workflows.yaml`.

The runtime also supports a direct Flow model slug through
`global.integrations.looper.flow.model_names`. The built-in default is
`vllm-sr/flow`. Direct Flow calls evaluate only decisions with
`algorithm.type=workflows`; they do not silently fall back to normal single-model
routes.

## Key Advantages

- Exposes a multi-step agent workflow as one model name: `vllm-sr/flow`.
- Keeps worker boundaries explicit: dynamic planners may only use the decision's
  `modelRefs`.
- Supports both static role plans and dynamic planner-generated workflows.
- Records a Flow trace with plan, worker steps, responses, failed models, and
  usage.

## What Problem Does It Solve?

Some requests need orchestration rather than a one-step route decision: split the
task, ask multiple workers for targeted work, verify or reconcile the outputs,
and return one final answer through the same chat completions API. `workflows`
makes that orchestration a router-owned policy while keeping the public model
surface as small as `vllm-sr/flow`.

## When to Use

- A route should expose a single model name but run a bounded micro-agent flow.
- The worker pool should come from the decision's `modelRefs`.
- You want static low-latency templates for predictable tasks.
- You want dynamic planner-generated workflows for harder reasoning, coding, or
  verification tasks.

## Configuration

Register the direct model slug:

```yaml
global:
  integrations:
    looper:
      endpoint: http://localhost:8899/v1/chat/completions
      max_response_bytes_mb: 32 # optional; caps a single upstream response body (default 32 MiB)
      flow:
        model_names:
          - vllm-sr/flow
        state:
          store_backend: file
          ttl_seconds: 1800
          file:
            directory: .vllm-sr/flow-state
```

Configure a dynamic Flow decision:

```yaml
routing:
  decisions:
    - name: coding_flow
      output_contract: Preserve any explicit output format exactly.
      modelRefs:
        - model: openrouter/gemini-pro
        - model: openrouter/deepseek
        - model: qwen/qwen3.6-rocm
      algorithm:
        type: workflows
        workflows:
          mode: dynamic
          planner:
            model: qwen-coordinator
            max_completion_tokens: 2048
          max_steps: 6
          max_parallel: 3
          round_timeout_seconds: 90
          min_successful_responses: 2
          on_error: skip
```

`output_contract` is decision-scoped prompt text. Use it for benchmark or
application format requirements that should apply across static Flow, dynamic
Flow, Fusion, and ReMoM instead of hard-coding task-specific prompts into an
algorithm. Use `output_contract_spec` for typed router-executable normalization
and post-processing such as choice extraction, terminal-action JSON
normalization, or reference dereferencing. Extraction defaults to exact
`content` matching; use `extract.sources` or `extract.mode: json_object` only
when the decision explicitly permits a wider parser.

The planner model is a control-plane model. It does not need to appear in
`modelRefs`. Worker calls are constrained to `modelRefs`; if the planner names a
model outside that list, the executor rejects the plan.

Static mode uses an explicit role plan. Each role model must be in the
decision's `modelRefs`.

```yaml
routing:
  decisions:
    - name: static_flow
      modelRefs:
        - model: qwen-worker
        - model: deepseek-worker
      algorithm:
        type: workflows
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
          on_error: skip
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_names` | list[string] | `["vllm-sr/flow"]` | Direct request model slugs that trigger Flow execution |
| `state.store_backend` | string | `file` | Pending tool-call workflow state backend: `memory`, `file`, or `redis` |
| `state.ttl_seconds` | int | `1800` | TTL for pending tool-call workflow state |
| `mode` | string | `static` | `static` role execution or `dynamic` planner-generated execution |
| `template` | string | `micro_agent` | Static workflow template name |
| `roles` | list[object] | required for static | Ordered static roles, each with `name`, `models`, optional `prompt`, and optional `access_list` of earlier role ids or agent ids |
| `final.model` | string | first worker response | Optional static final synthesis model from `modelRefs` |
| `final.prompt` | string | built-in synthesis prompt | Optional static final synthesis instruction |
| `planner.model` | string | required for dynamic | Control-plane model used to generate the workflow plan |
| `planner.max_completion_tokens` | int | `2048` | Max completion tokens for the planner JSON plan only |
| `max_steps` | int | `3` | Maximum workflow steps accepted from the planner |
| `max_parallel` | int | `2` | Maximum worker models per step |
| `max_completion_tokens` | int | request default | Max completion tokens for worker and final synthesis calls |
| `round_timeout_seconds` | int | unset | Maximum seconds to wait for each workflow step or final synthesis |
| `min_successful_responses` | int | all models | Continue a parallel step once this many workers succeed |
| `temperature` | float | request default | Temperature for planner, worker, and synthesis calls |
| `include_intermediate_responses` | bool | `true` | Include Flow plan and worker outputs in the response trace |
| `on_error` | string | `fail` | `fail` on worker error or `skip` failed workers when at least one worker succeeds |

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

Each worker has its own message history. A later step's `access_list` exposes
only prior step or prior agent outputs, not another worker's raw tool calls or
tool-result trajectory. Omitting `access_list` exposes all earlier step outputs;
setting it to `[]` isolates the step from prior outputs. Use a role id such as
`solver` to expose all outputs from that role, or an agent id such as
`solver:1:deepseek-worker` to expose only one worker from a parallel role. The
same agent id is emitted as `flow.steps[].responses[].agent_id` when
`include_intermediate_responses` is enabled.

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
