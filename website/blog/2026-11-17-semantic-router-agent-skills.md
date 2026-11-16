---
slug: semantic-router-agent-skills
title: "Agent skills for configuring and evaluating vLLM Semantic Router"
description: "A focused skill collection for serving the Router, understanding its decision model, configuring recipes, diagnosing routes, and comparing measured runs."
authors: [anupsharma]
tags: [agents, routing, evaluation, benchmarking, configuration, semantic-router]
---

A request reaches vLLM Semantic Router through a public entrypoint. The Router
evaluates signals, chooses a decision and algorithm, selects a backend, and
applies recipe-scoped plugins. A useful assistant must understand where in that
chain an observation came from. A healthy process does not prove a request took
the intended route; an improved benchmark score does not prove two runs were
comparable.

We have started publishing focused [agent skills](/docs/installation/agent-skills)
for those jobs. They give a coding assistant a short workflow, point to the
current commands and contracts, and say which evidence to preserve. The
[llm-d team's skills article](https://llm-d.ai/blog/skills-for-accelerated-llm-d-development-evaluation-and-configuration)
inspired this effort. Its examples show how reusable skills can connect
configuration, deployment, and evaluation. Here, the unit of work follows the
Semantic Router's entrypoint, recipe, decision model, and measured request.

<!-- truncate -->

## Choose the question first

The [operations skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md)
remains the broad guide for installation, upgrades, rollback, backup, and
teardown. The new skills answer narrower questions:

| Question | Skill | Result to review |
| --- | --- | --- |
| How do I start the local Router or decision model? | [Serve](https://vllm-sr.ai/install/agent/vllm-sr-serve/SKILL.md) | Runtime status, readiness, and a bounded probe |
| Which model answers a routing signal? | [Decision model](https://vllm-sr.ai/install/agent/vllm-sr-decision-model/SKILL.md) | Question-to-deployment map and traced answer |
| How should this workload enter a recipe? | [Configure recipe](https://vllm-sr.ai/install/agent/vllm-sr-configure-recipe/SKILL.md) | Baseline, candidate diff, validation, and plan |
| Where did this request go wrong? | [Diagnose routing](https://vllm-sr.ai/install/agent/vllm-sr-diagnose-routing/SKILL.md) | Preview, routed probe, and first failing boundary |
| What happened on a frozen workload? | [Benchmark](https://vllm-sr.ai/install/agent/vllm-sr-benchmark/SKILL.md) | Manifest, run ID, coverage, and report |
| Is candidate B better than A? | [Compare configurations](https://vllm-sr.ai/install/agent/vllm-sr-compare-configurations/SKILL.md) | Compatible paired runs, uncertainty, and cost basis |
| What should a recipe maintainer change next? | [Tune recipe](https://vllm-sr.ai/install/agent/vllm-sr-tune-recipe/SKILL.md) | Diagnostic case, guard floors, candidate, and handoff |

These skills compose in a natural order, but an agent can start with the one
question the user has. A diagnosis may end at an unavailable decision-model
deployment; it should not invent a routing fix. A comparison may end with
insufficient paired coverage; it should not convert two illustrative answers
into a quality claim.

## From workload to a routed request

The configure-recipe skill starts from the live config and a workload brief.
Its [two-backend fixture](https://github.com/vllm-project/semantic-router/tree/main/tools/agent/skills/vllm-sr-configure-recipe/fixtures)
shows a small, reviewable edit: keep the listener and providers, add a `code`
recipe, and expose it as `example/code` while retaining `vllm-sr/auto` as the
default. The assistant validates both documents, inspects the diff, and plans
the replacement against a running management endpoint before suggesting an
apply. Placeholder backends in that fixture are deliberately not treated as
reachable services.

When a deployed request behaves unexpectedly, the diagnosis skill freezes the
request and active config identity. It uses route preview to inspect signal
evaluation and the selected decision, then sends a bounded probe through the
public listener to inspect actual delivery. Its [three-case walkthrough](https://vllm-sr.ai/install/agent/vllm-sr-diagnose-routing/SKILL.md#three-case-diagnostic-walkthrough)
separates an expected route, an unknown or fallback route, and an upstream
delivery failure. That distinction matters because each case points to a
different repair.

The decision-model skill answers a related but distinct question: which
deployment serves a model-backed signal, what model card and readiness should
be checked, and what answer appeared in the trace? The serve skill uses the
existing `vllm-sr serve` local-image flow for a local stack and records both
process readiness and a routed or System One probe.

## From a run to a defensible comparison

The benchmark skill prepares a workload manifest and freezes its case IDs,
profile, grader, target, and source digest before dispatch. Its
[two-case smoke fixture](https://vllm-sr.ai/install/agent/vllm-sr-benchmark/references/two-case-smoke.md)
is synthetic and public. Preparing it checks the data path without making a
model call. A real result still needs reachable Router and model backends,
reviewed call and cost bounds, and saved run IDs.

The comparison skill then checks that two runs used compatible cases,
request profiles, graders, and price assumptions. It reports paired, failed,
and missing denominators alongside uncertainty. Its
[worked example](https://vllm-sr.ai/install/agent/vllm-sr-compare-configurations/references/paired-evidence.md)
is illustrative: with only two cases, even a valid paired calculation cannot
support a broad quality or savings claim. The tune-recipe skill uses that
evidence to propose a bounded recipe change and hands the candidate back for
validation and a fresh before/after run.

## What we have verified

The [workflow guide](/docs/installation/agent-skills) records the exact
offline fixture digests and source commit used for the initial checks. The
canonical parser accepted both recipe documents, and the two-case dataset
prepared deterministically. The skill catalog, public mirrors, links, and
command references have repository checks. These are checks of the skill and
fixture contracts, not measurements of routing quality or performance.

The next evidence milestone is a live stack run: validate and plan a
candidate against an active configuration, capture preview and listener
probes, execute bounded benchmark runs for both configurations, and preserve
the paired comparison receipts. We will publish measured claims only with
those run IDs, coverage, versions, and cost assumptions available for review.

Each public `SKILL.md` is generated from a repository-authored source. That
keeps an installable copy available to an agent without a checkout while the
source, examples, and checks evolve with the Router. The
[skill catalog](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/README.md)
is the entrypoint for contributors who want to improve a workflow or add a
case we missed.
