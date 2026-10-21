---
title: Coding Agents Through the Router
description: "Design proposal for serving coding-agent harnesses through the Router, ordered by what the measurements support: control, cache-aware placement, outcome-driven model choice, and gated escalation."
created: 2026-10-21
status: Proposal
---

> **Status:** Proposal · **Created:** 2026-10-21 · **Epic:** [#&lt;epic&gt;](https://github.com/vllm-project/semantic-router/issues)

## Problem

Coding agents such as Claude Code, Codex CLI, OpenCode and mini-SWE-agent send the longest sessions a Router sees. A run resends its whole conversation on every call, reuses almost all of its prompt, delegates parts of the task to subagents, and is judged by one outcome at the end: the tests pass or they do not. Teams that want these agents on a chosen set of models, often self-hosted, need one place to decide which models agents may use, to keep a run coherent and cached, and to choose models from their own outcomes.

Today a Claude Code session through the Router does not survive contact with an OpenAI-format backend. In ten Terminal-Bench 2.0 runs of Claude Code 2.1.304 through a Router built from main to vLLM Chat Completions backends, eight ended with "API Error: 400 translation would lose tool_result.is_error" at the first shell command that exited nonzero, and a backend serving Qwen3.5 or later refused the first answered call with "System message must be at the beginning." The client protocol fixes that already landed for Claude Code and Codex CLI (#4189, #4215, #4141, #4227, #4234) show the same pattern: each harness release finds a field the Router does not yet carry.

This proposal states what is worth building for coding agents, in which order, and how each step is evaluated. It keeps the boundary set by [Agent-Aware Router Contracts](./agent-based-routing): the Router selects a logical model and never orchestrates the agent.

## Position

Per-request model switching is the weakest reason to put a Router in front of coding agents. It captures a modest share of the theoretical headroom on public agent benchmarks, needs signals the Router does not see, and throws away a cache that holds almost the whole prompt. Four things are worth more, in this order.

1. **Control.** One place to decide which models an organization's agents may use, to keep a run on one model, to accept what harnesses send, to resolve the model names a harness uses internally, and to scan and audit what agents send. This is why a team that cannot adopt hosted coding tools would deploy the Router, and it does not depend on any cost argument.
2. **Cache-aware placement.** Keep a session on the model and replica that hold its prefix, and give self-hosted backends a CPU tier for the KV they would otherwise evict.
3. **Outcome-driven model choice.** Choose the default model, and the model a subagent gets, from the organization's own run outcomes and realized cost, and choose again when models change.
4. **Escalation at the end of an attempt**, as an option only where the first model costs a small fraction of the strong one: judge the finished attempt and restart on a stronger model when the expected rescue is worth it.

## Evidence

Sources, chosen for provenance. Production session traces: SemiAnalysis cc-traces-weka-with-subagents (Apache-2.0, 336 Claude Code sessions with token counts and 64-token KV block hashes, no text). Multi-model trajectories: Terminal-Bench 2.0 leaderboard trajectories (Apache-2.0), TwinRouterBench's static track (Apache-2.0), OpenHands trajectories from nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0), SWE-Router's SWE-Smith and SWE-bench Verified trajectories (arXiv 2607.00053) and mini-SWE-agent runs from the public SWE-bench Verified submissions, both used for aggregate results only. Public leaderboard rows from Terminal-Bench 4.0. Router replay of recorded runs and live Claude Code runs, both ours. Every number is a single run unless stated, and the research issues #&lt;R1&gt; (prefix reuse), #&lt;R2&gt; (CPU KV tier and placement) and #&lt;R3&gt; (escalation) carry the code and data behind them. The evaluation lanes below fetch only data whose license allows it and rerun the Terminal-Bench 2.0 numbers from it. The SWE-Router and SWE-bench trajectories are local inputs.

**What the Router does with agent traffic on main.**

- Tool-call agents are held on their first model for the whole run (343 of 388 replayed requests). Text-protocol agents that send command output as user messages are not recognized as tool loops: under `scope: conversation` the Router switched models mid-run 62 times in 30 replayed mini-SWE-agent runs when a signal on the latest output changed the decision, `min_turns_before_switch` did not stop it, and `scope: session` held every run on one model. Replaying 30 Terminal-Bench 2.0 runs the same way, 15 switched mid-run under conversation scope and none under session scope.
- Claude Code sends `x-claude-code-session-id` on every request, with the same value for the main agent and its subagents, and `x-claude-code-agent-id` only on subagent requests. Protection keyed on the session held both subagents of a run on the parent's model when their decision chose another, and protection keyed on `x-session-id`, which Claude Code never sends, held nothing.
- Every Claude Code session starts with rejected calls (`thinking.display: updates` and `tools.defer_loading`), and the run then drops the display setting.
- Toward OpenAI-format backends, a failed tool result is rejected (`tool_result.is_error`, strict by design since #4656) and Claude Code's mid-conversation system messages reach the backend in a position many open-model chat templates reject. Rendered through 21 published chat templates, a system message after a user turn raises in Qwen3.5, Qwen3.6, Qwen3.8, Gemma 3 and Mistral 7B v0.3 and is silently dropped by gpt-oss, MiniMax-M2 and Nemotron Nano 9B v2. Claude Code sends these messages throughout a session: its environment section after the first user turn, and notices after the turn they describe ("changed on disk since you last read it", "[Truncated: PARTIAL view ...]", "# Environment update").
- A vLLM server cannot be declared as a Messages backend (`provider: vllm` refuses `api_format: anthropic`), and a decision that enables reasoning for an `enabled`-mode family fails its own request on a Messages backend because enabled thinking there needs a token budget.
- Router Learning adaptation learns from outcomes (500 of 500 reached the ledger) but prices models by list price with a penalty capped at 0.04. Offered gpt-5-mini, deepseek-v3.2 and gpt-5.2 on SWE-bench Verified it converged to gpt-5.2 (0.662 at $0.140 per task, where gpt-5.2 alone is 0.742 at $0.258). Its learned state lives in process memory and restarts cold.

With the compatibility changes listed under Delivery, the same ten tasks ran through Chat Completions backends to a stock Qwen3.5 template with no API error: 274 requests carried one to five mid-conversation system messages and 189 carried a failed tool result, and the backend refused none.

**Cache.** In the SemiAnalysis sessions, 98.0% of a main agent's prompt tokens are a prefix of the same session's earlier calls. A subagent's first call shares 9% of its prompt with the parent's session and 63% with an earlier sibling, and subagents carry 10% of the tokens. Replaying those sessions against Qwen3-14B on one MI300X with WEKA's kv-cache-tester, vLLM with LMCache's CPU tier carried 2.2 times the input tokens of the GPU prefix cache alone (8.75M against 3.98M, and 2.1 times on a second run) under the same time-to-first-token limit, because the sessions' working set (1.03M tokens) exceeds the GPU KV pool (801,376 tokens). Across two replicas, keeping each session on one replica after a load-aware first placement raised prefix hits from about 80% to 90% and cut median time to first token from 1.6 s to 1.0 s when the sessions fit, while a plain hash of the session id did worse than round robin under load.

**Cross-model KV handoff.** The published Qwen3-14B to Qwen3-32B mapper is at KL 0.036 to the target's own cache on chat and 0.14 to 0.18 on OpenHands trajectories at 8k to 32k tokens, four to five times further on agent traffic. It still beats keeping only the last 2,048 tokens of history.

**Model choice.** The default model flips by workload: deepseek-v3.2 beats gpt-5-mini on SWE-bench Verified (0.624 at $0.045 against 0.556 at $0.033) and loses on SWE-Smith. List price misleads: on the Terminal-Bench 4.0 leaderboard, Claude Code with Sonnet 5.5 costs $22.22 per trial against $14.30 with Opus 5.5 and solves less (0.618 against 0.648). Online, two cheap models 0.07 apart did not separate reliably within 250 runs of Router Learning.

**Escalation.** Deciding from the task text does not work: a judge reading the issue or the first one to four agent actions ranks tasks a stronger model would rescue at AUROC 0.585 to 0.648, no better than ranking by predicted cost. Judging the finished attempt works if the judge sees the final diff (d3 and d3-turbo predict the cheap agent's failure at AUROC 0.735 to 0.755 with it, 0.668 to 0.709 without). It pays in a narrow band if at all: on the 87 Terminal-Bench 2.0 tasks both models attempted, escalating deepseek-v3.2 to Claude Opus 4.6 gains $0.10 and $0.16 of utility per task over the best single model at $2 and $5 per resolved task, with a judge at AUROC 0.856, but the 95% intervals ([-0.03, +0.24] and [-0.03, +0.33]) include zero, and escalating from gpt-5-mini or to Gemini 3 Pro loses. A deployment only sees the strong model's outcome on tasks it escalated, so learning the threshold costs most of the gain ($0.004 to $0.027 per task at $2 per resolved task over a 500-task stream, negative elsewhere). Per-call tier routing is not the target either: zero-shot d3-turbo scores 56.83 on TwinRouterBench, level with the best rule-based router the TwinRouterBench paper reports (56.96) and below always choosing the low tier (63.68).

**Subagents.** Whether a subagent on a cheaper model keeps a run's success is not in any public data. Our live attempt could not answer it: in 30 Claude Code runs on Terminal-Bench 2.0 with Qwen3.5-27B or Qwen3.8-27B as the main agent, none called the subagent tool.

## Design

### Control

- **One model per agent run, and per subagent run.** Recommend `scope: session` for coding-agent entrypoints, so text-protocol agents are held like tool-call agents. Add an optional agent identity header to protection (`x-claude-code-agent-id` for Claude Code), so each subagent run is held on the model chosen for its first request instead of on its parent's. A policy switch, such as a refusal fallback, stays visible in the run's records. When a switch inside a session pays is #4080's question, and until it shows one for coding agents the recommended configuration holds the run.
- **Accept what harnesses send.** Accept `thinking.display: updates` and pass it to Messages backends. Let operators accept the `tool_result.is_error` loss toward Chat and Responses backends by name, with a protocol warning, while every other loss keeps failing closed. Send mid-conversation system messages to Chat and Responses backends as user text in the same place, so no template refuses them and no notice moves away from the turn it describes. Let a vLLM server be declared as a Messages backend, and send a budgetless enabled reasoning mode to Messages backends as adaptive thinking at the selected effort. Keep rejecting `tools.defer_loading`, which needs the backend's tool search, and document that the harness retries without it.
- **Resolve the harness's model names.** Harnesses name models for subagents and fallbacks (`sonnet`, `haiku`, `opus`, versioned ids). The coding-agent recipe lists every name the harness can send in its entrypoints' `model_names`, so a delegation does not fail on an unknown model.
- **Scan what agents send.** PII and jailbreak signals already cover prompts, and #3624 covers tool results. Add credential and private-key detection for prompts and tool results, evaluated on real leaked-secret corpora before it is claimed.
- **Audit without content.** Keep replay content-minimized, and record per run the model or models used, cost, scan results, and the outcome when the harness reports one.

### Cache-aware placement

- **Sessions stay where their cache is.** Use the cache-affinity estimator (#4188) with session identity for coding-agent entrypoints, so a session's requests reach the same model. Behind a model, place replicas with a scheduler that weighs prefix cache and load, as llm-d's endpoint picker does, rather than with `lb_policy`. A session-sticky `lb_policy` is not proposed: a hash alone did worse than round robin under load, and a load-aware version would have to behave the same in the in-process balancer and the Envoy cluster.
- **A CPU tier for self-hosted backends.** Document LMCache's CPU tier for vLLM backends that serve agent sessions, sized from the session working set.
- **Cross-model KV handoff stays off for coding agents** until a mapper trained on agent transcripts closes the gap above. The coordinator's `max_transfer_turn` bound remains the guard where it is on.

### Outcome-driven model choice

- **Outcomes per run, not per request.** A harness reports one outcome per run (tests passed, a reviewer verdict) together with its realized cost, linked to the run's requests by session id, or the Router grades a finished task itself as #5064 proposes.
- **Offline first.** Recipe learning groups replay records into runs and reports, per decision and model, the success rate with an interval, the realized cost per run, and the expected value at a stated value per success. It proposes a default model as a recipe patch for review and keeps runs that used more than one model apart. Nothing changes on the request path.
- **Subagents by role, from lineage.** When #3379 carries a request's delegated role, a recipe can give a role its own model, and the same offline report scores it from run outcomes. It becomes a default only after a live run with a main agent that delegates shows that a cheaper subagent keeps the run's success.
- **Online through #3412.** Request-time selection by expected lifecycle cost under a success target is #3412's. Our replay and leaderboard measurements are evidence for it: list price is not the cost that matters for agents.

### Escalation at the end of an attempt (optional)

- **Contract.** When the cheap agent declares the task done, the harness sends the task, the final diff and the agent's actions with their outputs. The Router answers through the System One API with a fixed question set (will the patch resolve the task, would a much stronger model starting from scratch resolve it) and returns a verdict. It is the attempt-level counterpart of the per-request residual-failure classifier in #3282, and the verdict can feed Router Learning the way #5064 proposes for finished tasks.
- **Rule.** Escalate when the value of a resolved task times the calibrated probability of a rescue exceeds the predicted strong-model cost. Calibration needs both models' outcomes, so the loop escalates a small random share of tasks regardless of the verdict and fits only on that sample. Restart the task on the strong model rather than handing it the cheap transcript.
- **When to enable.** Only where the evaluation shows the band: a first model that costs a small fraction of the strong one, and a judge that is accurate on that model's attempts.

## Evaluation

Four lanes, each reproducible from the repository on licensed data:

1. **Counterfactual policy evaluation** on recorded multi-model trajectories: for any policy, resolve rate at fixed budgets against the static hull of single models, utility at stated values per resolved task, judge AUROC overall and within task, and the gain when thresholds are calibrated on the first outcomes of a stream. No inference.
2. **Recorded-run replay through the Router**, with a backend that answers from the recording, in both wire forms. It shows what the Router holds, switches, records and learns, and streams a benchmark through Router Learning with an outcome posted after each run.
3. **Live coding-agent runs**: an unmodified harness, pinned by version, on Terminal-Bench tasks through the Router, reporting the verifier's reward, API errors, subagents, and the Router's records joined by session. This is the lane that found the compatibility failures above, and the one that answers what recordings cannot: subagents on a cheaper model, restart against continuation, end-to-end latency and cache effects.
4. **Cache and KV**: production session traces replayed against a vLLM backend with and without a CPU tier, and the KV handoff measured as KL to the target's own cache on agent trajectories (#2976).

## Delivery

| Step | Work | Tracking |
| --- | --- | --- |
| 1 | Harness compatibility: agent identity in protection, `thinking.display: updates`, the opt-in `tool_result.is_error` loss, mid-conversation system messages, vLLM Messages backends, budgetless reasoning on Messages backends | Epic children |
| 2 | Coding-agent guide and recipe: session scope, harness model ids in `model_names`, cache affinity on session identity, CPU KV tier, cache-losing events | Epic child |
| 3 | Run-level outcomes and the offline model-choice report in recipe learning | Epic child |
| 4 | Evaluation lanes 1 to 3 in the repository, and lane 4 for agent trajectories | Epic children, #2976 |
| 5 | Live subagent experiment with a main agent that delegates | Epic child |
| 6 | Escalation contract and a reference harness wrapper, behind the evaluation gate | Later |
| 7 | Credential and private-key scanning, with its evaluation | Later |
| 8 | Router Learning cost evidence for #3412 | #3412 |

## Scope and non-goals

This proposal does not:

- orchestrate, host or invoke agents, or add an agent catalog to Router configuration.
- switch models inside a run by default, or hand a cheap model's transcript to a strong model before that is measured.
- enable cross-model KV handoff for coding-agent entrypoints.
- claim a routing gain, scan accuracy or escalation benefit that the evaluation lanes have not shown on licensed data.

## Related work

- [Agent-Aware Router Contracts](./agent-based-routing), Epic #2994, and #3379 with #4849 for delegated-role and lineage facts.
- Epic #2973 for long-session state and model switching, #4080 for switch calibration, and #2984 for context optimization.
- Epic #2238 and #3412 for the offline-to-online recipe lifecycle and lifecycle-cost selection, #3503 for a Router Learning benchmark, and #5064 for task-level outcome verification.
- #3282 for a residual-failure escalation classifier.
- Epic #2976 and #4735 for cross-model KV reuse.
- Epic #4241, #4153, #4259 and #4964 for agent workflows and session replay in sr-bench, and #3410 for a real-client smoke check.

## Open questions

- Does a subagent on a cheaper model keep the run's success when the main agent delegates?
- Does continuing on the strong model with the cheap transcript beat restarting?
- Can a mapper trained on agent transcripts bring the KV handoff close to its chat numbers?
- Do standalone deployments without llm-d serve agents from several replicas often enough to justify a load-aware sticky `lb_policy`?
- How often does an organization's best default model change over time?
