# PL-0046: Coding Agents Through the Router (Epic #5378)

## Goal

Make the Router a dependable place to run coding-agent harnesses on a chosen set
of models, in the order the measurements support: control, cache-aware
placement, outcome-driven model choice, and escalation at the end of an attempt
as an option. The Router selects logical models and never orchestrates the
agent.

## Scope

- Proposal PR: design document, proposals index, sidebar, and this execution
  plan.
- Track the Epic children: harness compatibility, the coding-agent sections of
  the agent harness guide, run-level outcomes in recipe learning, the evaluation
  lanes, the research issues behind the proposal's numbers, and the live
  subagent experiment.
- Coordinate run and subagent stability with #2973 and #4080, delegated-role
  facts with #3379, lifecycle-cost selection with #3412, cross-model KV with
  #2976, and the coding-agent session replay with #4241.

## Non-Goals

- Router-native agent invocation, hosting, or an agent catalog in Router
  configuration.
- Model switches inside a run by default, or handing a cheap model's transcript
  to a strong model before that is measured.
- Cross-model KV handoff for coding-agent entrypoints.
- Online lifecycle-cost selection, which stays with #3412.

## Exit Criteria

- [ ] Proposal merged at `website/docs/proposals/coding-agents.md`.
- [ ] Claude Code and Codex CLI complete Terminal-Bench runs through the Router
  to Chat, Responses and Messages backends with no Router-caused API error.
- [ ] A run and each of its subagent runs stay on one model under the
  recommended configuration.
- [ ] Recipe learning reports success rate and realized cost per run.
- [ ] The evaluation lanes run from the repository on licensed data.
- [ ] Open questions in the proposal resolved or explicitly deferred.

## Task List

- [x] `PROP-01` Draft the proposal with its evidence and delivery order.
- [x] `PROP-02` Add the proposal to the index and the website sidebar.
- [x] `PROP-03` Add execution plan PL-0046.
- [ ] `PROP-04` Maintainer review of the order and the open questions.
- [ ] `STEP-1` Harness compatibility: agent identity in protection,
  `thinking.display: updates`, the opt-in `tool_result.is_error` loss,
  mid-conversation system messages, vLLM Messages backends, budgetless
  reasoning on Messages backends.
- [ ] `STEP-2` Coding-agent sections in the agent harness guide.
- [ ] `STEP-3` Run-level outcomes and the offline model-choice report in recipe
  learning.
- [ ] `STEP-4` Evaluation lanes 1 to 3, and lane 4 for agent trajectories under
  #2976.
- [ ] `STEP-5` Live subagent experiment with a main agent that delegates.
- [ ] `STEP-6` Escalation contract and a reference harness wrapper, behind the
  evaluation gate.
- [ ] `STEP-7` Credential and private-key scanning, with its evaluation.
- [ ] `STEP-8` Router Learning cost evidence for #3412.

## Next Action

Open the Epic children with their PRs, starting with harness compatibility.
Steps 6 and 7 wait for the evaluation lanes.

## Operating Rules

- One Epic child per PR, and compatibility fixes ship apart from design changes.
- A routing default changes only after a lane shows the gain on licensed data.
- Every harness field the Router accepts or rejects is documented, and the live
  lane records the harness version it ran.
- Update the routing-surface catalog and public docs when a step ships
  user-visible behavior.

## Related Docs

- [Coding Agents Through the Router proposal](../../../../website/docs/proposals/coding-agents.md)
- [Agent-Aware Router Contracts proposal](../../../../website/docs/proposals/agent-based-routing.md)
- [Success-Constrained Model Selection proposal](../../../../website/docs/proposals/success-constrained-selection.md)
- [Connect an agent harness](../../../../website/docs/installation/agent-harness.md)
- [Epic #5378](https://github.com/vllm-project/semantic-router/issues/5378)
- [Epic #2973](https://github.com/vllm-project/semantic-router/issues/2973)
- [Feature #3379](https://github.com/vllm-project/semantic-router/issues/3379)
- [Feature #3412](https://github.com/vllm-project/semantic-router/issues/3412)
