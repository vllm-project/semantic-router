# Tools

## Overview

`tools` is the route-local plugin for the request's tools. Its `policy`
allows, blocks or removes tools by name. Its `select` block asks the decision
model which tools the request needs, among the request's own tools
(`mode: filter`) or the tools of a catalog (`mode: add`).

## Key Advantages

- Keeps tool policy attached to the matched route.
- One decision-model call per request chooses the tools: `tool_need` (does the
  request need a tool) and `tool_rank` (which ones), asked together.
- Filter mode keeps the request's tool order, so a provider's prompt cache
  keeps its prefix.

## What Problem Does It Solve?

Tool behavior is part of route policy. Some routes must strip tools entirely,
some pass them through, and some should send only the tools a request needs:
large tool lists cost tokens and confuse models. The `tools` plugin makes that
route-local contract explicit, and the decision model, which already answers
the route's signals, chooses the tools.

## When to Use

- a route should disable all tools
- a route should restrict tool access with explicit allow and block lists
- a route should send only the tools a request needs, or add the relevant
  tools of a catalog
- a privacy route should keep tool history available for routing but omit
  prior tool calls and results from the selected model request

## Configuration

Add the plugin under `recipes[].decisions[].plugins`:

```yaml
plugins:
  - type: tools
    configuration:
      enabled: true
      policy:                         # deterministic, runs first
        mode: filtered                # passthrough | filtered | none
        allow_tools: [docs.search, tickets.lookup]
        block_tools: [admin.delete]
        strip_tool_history: false
      select:                         # the decision model chooses inside the tools the policy allows
        mode: add                     # filter | add
        tools_db_path: config/tools_db.json
        match:
          judge: {preset: tool_rank, top_k: 3, min_ratio: 0.1, max_candidates: 255}
        need_threshold: 0.5           # P(tool_need) below this: no tools are kept or added
        preserve_count: 0             # filter mode: the request's first tools are always kept
        on_error: skip                # add: skip | fail; filter: keep | strip | fail
        input: {history: false}
        timeout_ms: 1000
```

### Policy

- `passthrough` (the default) leaves the request's tools as they are.
- `filtered` keeps the tools `allow_tools` names, when it names any, and
  removes the tools `block_tools` names. A Responses hosted tool goes by its
  type, such as `image_generation`. A `tool_choice` that names a removed
  tool, a hosted tool included, is removed with it, and the model chooses
  among the tools left.
- `none` removes every tool, a Responses hosted tool such as
  `image_generation` included, and the request's `tool_choice`. With
  `strip_tool_history: true` it also removes prior assistant tool calls and
  tool results from the provider-bound request. The policy runs after signal
  and decision evaluation, so it never changes which route matched.
- `trusted_facts` gates tool use on operator policy, gateway attestation and
  fresh runtime availability (the decision's catalog, when its selection adds
  from one). It is off unless enabled.

A decision whose algorithm makes several model calls, such as `escalate`,
`fusion` or `workflows`, applies the policy and the selection once to the
request, and each call it makes carries at most the tools left. A workflow
that resumes after a client tool call holds the calls it builds from the
paused request to the policy in force when it resumes. With `trusted_facts`
on, the selection of such a decision adds no catalog tools.

### Selection

The selection runs when the request leaves tool use to the model: no
`tool_choice`, or `auto`. One call to the deployment that answers the judge
(the decision model unless `match.judge.deployment` names another) asks two
questions about the current user turn, and the earlier user turns with
`input: {history: true}`:

- `tool_need`, a yes/no question: does answering the request need a tool?
- the `match` judge, by default `tool_rank`: a choice whose options are the
  candidate tools, each described by its description. Its keep rules
  `top_k`, `min_ratio` and `min_score` decide which tools it keeps.

The candidates and the result depend on the mode:

| Mode | Candidates | Result |
| --- | --- | --- |
| `filter` | The request's tools the policy kept | The tools the judge keeps and the first `preserve_count` tools, in the request's order |
| `add` | The catalog's tools the policy allows that the request does not carry | The request's tools, then the tools the judge keeps, best first |

- When P(`tool_need`) is below `need_threshold`, the judge's tools are neither
  kept nor added. Filter mode then keeps only the first `preserve_count`
  tools; a request left without tools also loses its `tool_choice`.
- `on_error` is what a failed or late call does: filter mode keeps the
  request's tools (`keep`), removes them (`strip`) or refuses the request
  (`fail`); add mode adds none (`skip`) or refuses the request (`fail`). A
  refused request gets `503` with the code
  [`tool_selection_failed`](../../api/router.md#routing-errors).
- One question takes at most 255 options, so the judge reads at most
  `match.judge.max_candidates` tools (255 by default): the first as the
  request or the catalog lists them. More tools are never refused; a
  `recall: {method: bm25, top_k: ...}` stage instead narrows them to the ones
  whose descriptions share the most terms with the request. Recall only
  narrows, so a `match` block without a judge fails to load. Each truncation is
  a `match.judge.truncated` event on the request's trace span and a count in
  `vsr_match_judge_truncated_total`.
- `timeout_ms` bounds the call, up to 60000. Without it a systemone
  deployment's own `timeout_ms` applies, else 1000. The judge's own
  `on_error` and `timeout_ms` are not accepted: the selection's govern.

The catalog of add mode is a JSON list of entries
`{"tool": {"type": "function", "function": {...}}, "description": "..."}`;
`description` is what the decision model reads to choose the tool, the
function's description when it is absent. A relative `tools_db_path` is read
from the Router's working directory. The load checks a catalog it can read,
of any size: unique names (`duplicate_tool_name`) and at least one tool the
policy allows. The Router reads the catalog again when it builds the
configuration it serves, and refuses one it cannot read.

Each selection is recorded in the request's
[router replay](./router-replay) record: the tools it kept, P(`tool_need`)
and its outcome. Its judge is counted in `vsr_match_judge_total{feature="tools"}`.

### Migrating

The `tool_selection` plugin is the `select` block, `global.integrations.tools`
is removed, and the policy fields moved under `policy`. A configuration that
still uses these forms fails at load with a one-line hint, and
`vllm-sr config migrate` rewrites it. Embedding similarity thresholds
(`similarity_threshold`, `relevance_threshold`) have no judge equivalent:
migrate leaves a note, and `min_ratio` or `min_score` can be fitted to the
decision model when the selections need one.

Tool selection controls what reaches the model; it does not authorize tool
execution. Enforce permissions at the tool service and treat tool schemas and
results as provider-bound content. Complete examples:
[`config/fragments/plugin/tools/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/tools).
