# PL-0035 Router Flow Workflows

## Goal

Implement Router Flow as vLLM Semantic Router single-model multi-agent
orchestration: one OpenAI-compatible model slug,
`vllm-sr/flow`, backed by configurable `workflows` algorithms over explicit
worker pools. The first implementation excludes coordinator training and focuses
on runtime orchestration, config, docs, local/AMD validation, eval assets, and a
publishable blog.

## Scope

- Add a special direct model slug, `vllm-sr/flow`, parallel to
  `vllm-sr/fusion`.
- Add `algorithm.type: workflows` with `mode: static` and `mode: dynamic`.
- Keep worker selection inside each decision's `modelRefs`; do not add a
  separate `source: modelRefs` field.
- Keep dynamic planner selection explicit through
  `algorithm.workflows.planner.model`; the model must be configured in the
  normal provider/model registry.
- Implement a runtime planner -> worker steps -> final synthesis loop without
  training a coordinator model.
- Validate model recursion and plan boundaries so Flow cannot call Flow as its
  own planner and dynamic plans cannot execute workers outside `modelRefs`.
- Update Go config, extproc, looper, DSL compile/decompile, Python CLI schema,
  dashboard surfaces, docs, examples, and tests.
- Add a multi-agent API eval harness and AMD/OpenRouter validation recipe without
  committing secrets.
- Draft the public blog in `vllm-project/vllm-project.github.io`.

## Exit Criteria

- `model: "vllm-sr/flow"` executes only matching workflow decisions.
- `algorithm.type: workflows` works from canonical YAML, DSL, Python CLI schema,
  and dashboard topology/builder surfaces.
- Static mode executes an explicit ordered role plan over decision `modelRefs`
  without requiring a planner.
- Dynamic mode calls `planner.model` to produce a bounded workflow plan
  serialized as JSON for router-side validation, validates that worker/final
  models are in `modelRefs`, executes bounded parallel worker steps, and
  synthesizes a final answer.
- Dynamic mode fails validation when `planner.model` is missing.
- Flow response headers and traces expose algorithm, model calls, iterations,
  and optional intermediate plan/worker outputs.
- Direct Flow slugs live under `global.integrations.looper.flow.model_names`;
  workflow policy stays under `routing.decisions[].algorithm.workflows`.
- Coordinator training and RL optimization are explicitly deferred.
- Local gates pass for changed Go, Python, dashboard, docs, and config surfaces.
- AMD validation uses the repo AMD local serve path and records private evidence
  outside public docs.
- Publishable eval results are produced from EvalScope or benchmark-native
  reports through `bench/router_flow/real_eval/collect_evalscope_results.py`.
  The older one-prompt proxy harness remains development-only smoke data.

## Task List

- [x] Start from latest `origin/main` on a dedicated branch.
- [x] Research external multi-agent-as-model product and coordination references.
- [x] Add Go config surface for `global.integrations.looper.flow` and
  `algorithm.workflows`.
- [x] Add direct Flow routing in extproc and looper factory wiring.
- [x] Implement static role-plan and dynamic planner `WorkflowsLooper` execution.
- [x] Implement multi-agent function-calling workflow support: per-agent tool
  loops, access-list output visibility, private tool trajectories, pending
  state persistence, and tool-result routing back to the requesting agent.
- [x] Extend `access_list` from step-only visibility to step-or-agent
  visibility, with stable `flow.steps[].responses[].agent_id` trace ids for
  parallel worker outputs.
- [x] Add Python CLI schema and validator support.
- [x] Add DSL compile/decompile support for workflows.
- [x] Add dashboard DSL/topology/builder support.
- [x] Add reference config and tutorial docs.
- [x] Add multi-agent API proxy eval harness and AMD recipe.
- [x] Add EvalScope-backed real eval runner and report collector.
- [x] Fix the EvalScope LiveCodeBench stdin wrapper so sandboxed solutions using
  `sys.stdin.buffer` are scored correctly.
- [x] Run focused unit tests and repo-native local gates.
- [x] Run AMD proxy regression and collect Flow/single-model comparison data.
- [ ] Run the full aligned EvalScope matrix for auto/fusion/flow. Use bounded
  rows only as interim engineering evidence, keep sample counts visible, and
  promote to formal/full limits before treating the comparison as complete.
- [x] Write the public blog draft in the website repo.

## Next Action

Finish the aligned-core formal EvalScope matrix for `vllm-sr/auto`,
`vllm-sr/fusion`, and `vllm-sr/flow`: GPQA-D, HLE text-only, LiveCodeBench
Jan-Apr 2025, SciCode with background, AA-LCR, and MRCR 8-needle up to 128K.
Then use the collector to identify missing or underperforming cells, fix
router/Flow/Fusion behavior where needed, and only then expand to SWE,
TerminalBench, Tau3, LiveCodeBench Pro, CharXiv, and multimodal HLE adapters.

## Operating Rules

- Keep public configuration small: direct model slugs under global integration
  config, route policy under decision algorithms, worker pool under `modelRefs`.
- Do not commit API keys, AMD hostnames, private paths, or private validation
  logs into public docs.
- Treat external benchmark numbers as competitor claims unless reproduced by our
  harness.
- Prebuild suite-declared local sandbox images before sandboxed EvalScope rows
  so code benchmarks fail only on model behavior, not missing Docker images.
- Keep planner prompts/templates replaceable, but do not add training hooks in
  this implementation.
- If Flow runtime gaps remain after this plan, add indexed technical debt
  rather than leaving them in chat.

## Related Docs

- [Router Flow Workflows proposal](../../../website/docs/proposals/router-flow-workflows.md)
- [Router Flow tutorial](../../../website/docs/tutorials/algorithm/looper/workflows.md)
- [Fusion API plan](pl-0034-fusion-api.md)
