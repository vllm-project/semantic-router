---
name: vllm-sr-tune-recipe
description: Turn a diagnosed vLLM Semantic Router misroute into one tested recipe proposal. Use when a contributor needs to choose manual calibration or the recipe-lab loop, preserve guard floors and holdout discipline, and explain keep, revert or inconclusive evidence.
---

# Improve one recipe decision

Start with an observed misroute or measurable objective, not a preferred YAML
edit. Use the [routing-diagnosis skill](https://vllm-sr.ai/install/agent/vllm-sr-diagnose-routing/SKILL.md)
to locate the failing boundary and the [benchmark skill](https://vllm-sr.ai/install/agent/vllm-sr-benchmark/SKILL.md)
to freeze evidence. The existing
[recipe-tuning guide](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md)
owns manual signal/decision mechanics. The
[routing-calibration skill](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/routing-calibration/SKILL.md)
owns maintained-recipe live probes. The automated agent loop is tracked in
[#5497](https://github.com/vllm-project/semantic-router/issues/5497); use its
skill and ledger when it ships, without copying its commands or verdict logic
here.

## 1. Prove the change belongs to the recipe

Record active config hash/version, entrypoint, recipe, request fixture,
expected decision, traced signal answers and unknowns, chosen algorithm and
model, plugin path, upstream delivery, and affected case IDs. A backend
timeout, model-runtime load failure or missing response is not evidence that
a signal question or threshold should change. If the first failing boundary
is outside recipe policy, hand it to that owner.

Choose one coherent intervention from the evidence:

| Evidence | Candidate area | Check before editing |
| --- | --- | --- |
| Answer is unknown or late | Deployment, question capability, timeout, `on_unknown` | Model-runtime readiness and error reason |
| Answer is present but wrong for intended cases | Question wording, labels, predicate or calibrated model | Counterexamples and answer distribution |
| Correct signals select the wrong lane | Decision condition, priority, projection or fallback | Complete `eval_trace` and precedence |
| Correct lane selects a poor backend | Model refs, binding or algorithm | Backend eligibility, price, capacity and paired results |
| Selected backend answers, but response is blocked or rewritten | Recipe/decision plugin | Plugin outcome, guard floor and delivery receipt |

Use the installed schema and maintained recipe card; preserve unrelated
entrypoints, providers and global policy. A new signal or model binding may
load a model, change latency or alter guard behavior. State those effects in
the proposal.

## 2. Freeze evidence and choose the loop

Before proposing a value, freeze development cases, counterexamples, guard
floors for safety/privacy/reliability, cost and latency objectives, the current
recipe and a rollback point. Reserve a disjoint holdout by stable case/input
identity; a new seed or label does not make exposed cases unseen. Use at least
two eligible, reachable answer models for a model-selection claim.

- **Manual path:** follow the recipe-tuning guide and the repository-local
  routing-calibration skill for maintained probe manifests and live checks.
- **Automated path:** when #5497's `recipe-lab-loop` skill is merged, let its
  CLI and ledger own candidate submission, keep/revert decisions and replay
  checks. Link its run and ledger record; do not synthesize one here.
- **Insufficient evidence:** if signal answers, paired cases, prices, grader
  identity or model readiness are missing, name the exact missing check and
  defer a tuning verdict.

## 3. Test one candidate

Write a separate canonical candidate with one coherent policy change and a
reviewable diff. Validate and plan it through the existing
[configuration workflow](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md#change-a-running-stack).
For an authorized development stack, activate the candidate, verify its
actual hash, preview affected and counterexample routes, and probe delivery.
Use the same frozen development workload and request profile for before/after
sr-bench runs. For a paired verdict, follow the
[comparison skill](https://vllm-sr.ai/install/agent/vllm-sr-compare-configurations/SKILL.md).
Do not adapt thresholds from holdout results or silently change backend
budgets between runs.

Classify the result as **keep**, **revert** or **inconclusive** against the
prespecified objective and guard floors. Keep requires complete compatible
evidence and no guard-floor breach; revert when a floor or objective fails;
inconclusive means missing/weak evidence or confounding changes. Preserve the
old and candidate recipes, failed attempts, run IDs and limitations. A human
reviews any promotion to shared or production traffic; this skill does not
promote automatically.

The [maintained-recipe walkthrough](references/maintained-recipe-walkthrough.md)
shows the evidence handoff on the Balance recipe, including the points where
live calibration and the recipe-lab ledger are required.
