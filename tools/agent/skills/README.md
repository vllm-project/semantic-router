# vLLM Semantic Router skills

The [catalog](catalog.yaml) lists shipped skills, their audience, source,
distribution, prerequisites, outputs, and handoff point. A skill is a task
guide over maintained CLI, API, and test contracts; it does not define a new
Router behavior. Choose one by the task below. The development harness
[reports changed-path checks](../docs/README.md); it does not select a skill.

| Your task | Start here | First supported step |
| --- | --- | --- |
| Run a local Router stack or decision model with `vllm-sr serve` | [Serve](vllm-sr-serve/SKILL.md) | Inspect the existing stack and config, then choose Router or Engine mode. |
| Install or operate a Router stack beyond first start | [Operations](vllm-sr-agent-operations/SKILL.md) | Run its read-only preflight, then choose the maintained local or Kubernetes path. |
| Choose or inspect the decision model | [Operations: optimize](vllm-sr-agent-operations/references/optimization.md) | Measure the baseline and inspect the model-runtime inventory. |
| Prepare a Router configuration or recipe change | [Operations: configure](vllm-sr-agent-operations/references/configuration-loop.md) | Read the current canonical configuration and discover the installed schema. |
| Verify or diagnose a route | [Operations: verify](vllm-sr-agent-operations/references/route-verification.md) | Preview the route, then probe the full data path. |
| Evaluate quality, latency, and cost | [Operations: sr-bench](vllm-sr-agent-operations/references/sr-bench.md) | Freeze the workload and choose preview, live, or replay scope. |
| Calibrate a maintained recipe | [Routing calibration](routing-calibration/SKILL.md) | Validate the versioned probe manifest. |
| Change a request-visible Router contract | [Router contract change](router-contract-change/SKILL.md) | Read the affected change surfaces and nearest `AGENTS.md`. |
| Review issues, PRs, or releases | [Maintainer operations](maintainer-ops/SKILL.md) | Resolve the exact target and current revision. |
| Set up the vSR Claude Code plugin | [Plugin setup](../../../src/agent-harness/claude-code/skills/setup/SKILL.md) | Detect the machine and current runtime. |
| Tune the vSR Claude Code plugin | [Plugin calibration](../../../src/agent-harness/claude-code/skills/calibrate/SKILL.md) | Inspect its recorded decisions. |

The [public serve skill](https://vllm-sr.ai/install/agent/vllm-sr-serve/SKILL.md)
handles a focused first start. The [public operations skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md)
covers the broader lifecycle. Their authored sources are under
`tools/agent/skills/`; `make agent-skill-sync` generates their website copies.
The contributor skills are repository-local. The plugin skills ship inside
the Claude Code plugin and are specific to that harness. Install the public
skill using the [agent installation guide](https://vllm-sr.ai/docs/installation/agent).

For example, a request to “run vLLM-SR with my local Ollama model” starts at
Serve → preflight. The first checks read Docker access, Python, memory,
disk, ports, and existing stacks. The agent then follows the supported
`vllm-sr serve --config config.yaml` path and verifies a routed request.
It must not start an incomplete config or replace a running stack without a
reviewed handoff. The configuration-planning research in
[#4880](https://github.com/vllm-project/semantic-router/issues/4880) and
[#4972](https://github.com/vllm-project/semantic-router/issues/4972) covers
agent discovery of the read-and-plan API path.

## Proposed additions

These are planned under [the skills epic](https://github.com/vllm-project/semantic-router/issues/6386)
and do not yet have standalone skill files:

| Proposed skill | Bounded workflow | Current entry |
| --- | --- | --- |
| `vllm-sr-decision-model` | Choose a supported model/runtime, check required recipe signals, readiness and answers, then collect calibration evidence. | [Optimization reference](vllm-sr-agent-operations/references/optimization.md) |
| `vllm-sr-configure-recipe` | Turn workload goals into a reviewable canonical config proposal. | [Configuration reference](vllm-sr-agent-operations/references/configuration-loop.md) |
| `vllm-sr-diagnose-routing` | Explain a selected route or failure from real Router evidence. | [Verification reference](vllm-sr-agent-operations/references/route-verification.md) |
| `vllm-sr-benchmark` | Run a bounded, reproducible sr-bench evaluation. | [sr-bench reference](vllm-sr-agent-operations/references/sr-bench.md) |
| `vllm-sr-compare-configurations` | Compare paired runs and explain incomplete evidence. | [sr-bench reference](vllm-sr-agent-operations/references/sr-bench.md) |

The recipe-lab agent loop is already tracked in [#5497](https://github.com/vllm-project/semantic-router/issues/5497).
Link to that skill when it lands; do not create a competing tuning workflow.

## Authoring and publication contract

- Put one task, its trigger, prerequisites, evidence, stopping point, and
  failure path in each `SKILL.md`. Keep detailed command procedures in small
  references owned by the same workflow.
- Add each shipped skill to `catalog.yaml`. Its `source` must be the sole
  authored `SKILL.md`; a public copy needs an explicit `published` path and
  `public_url`. A plugin-bundled skill stays within its plugin directory.
- Public website skills are generated from their repository sources.
  Edit the source, run `make agent-skill-sync`, and commit both. Do not edit
  website mirrors directly.
- Use the installed CLI schema, Router API, and maintained docs as authority.
  Describe unsupported versions and environments; do not guess commands.
- Keep credentials out of prompts, files, arguments, and published evidence.
  Report the observable success criteria and ask before activation or
  destructive operations.
- Run `make agent-skill-check` for catalog and public-copy consistency, and
  `make harness-check` when changing harness code or publication checks.
