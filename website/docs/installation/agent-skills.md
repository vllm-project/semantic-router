---
title: Agent skill workflows
description: Choose a focused vLLM Semantic Router skill and follow reproducible configuration, evaluation, and diagnosis examples.
sidebar_position: 3
---

# Agent skill workflows

The [agent installation guide](agent) gives a prompt for the complete
operations lifecycle. These focused skills handle a narrower question. Each
public `SKILL.md` is generated from one repository-authored source and can be
loaded by a coding agent without a repository checkout.

| Goal | Start with | Evidence to keep |
| --- | --- | --- |
| Start a local Router or decision model | [Serve skill](https://vllm-sr.ai/install/agent/vllm-sr-serve/SKILL.md) | Config/artifact, platform, status, readiness and a routed or System One probe |
| Resolve model-backed signals | [Decision-model skill](https://vllm-sr.ai/install/agent/vllm-sr-decision-model/SKILL.md) | Question-to-deployment map, inventory, answer and trace |
| Propose a workload policy | [Configure-recipe skill](https://vllm-sr.ai/install/agent/vllm-sr-configure-recipe/SKILL.md) | Baseline, candidate diff, validate and plan receipts |
| Explain a wrong route or failed answer | [Routing-diagnosis skill](https://vllm-sr.ai/install/agent/vllm-sr-diagnose-routing/SKILL.md) | Frozen request, preview, probe, headers and first failing boundary |
| Measure a frozen workload | [Benchmark skill](https://vllm-sr.ai/install/agent/vllm-sr-benchmark/SKILL.md) | Dataset/plan digests, run ID, coverage and report |
| Judge an A/B result | [Comparison skill](https://vllm-sr.ai/install/agent/vllm-sr-compare-configurations/SKILL.md) | Compatible run IDs, paired cases, uncertainty and cost basis |
| Improve a maintained recipe | [Tune-recipe skill](https://vllm-sr.ai/install/agent/vllm-sr-tune-recipe/SKILL.md) | Diagnostic case, guard floors, candidate diff and before/after runs |

The broader [operations skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md)
covers installation, Kubernetes, upgrades, rollback, backup and teardown. The
[repository catalog](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/README.md)
also lists contributor and harness-specific skills.

## Example 1: configuration to route preview

The configure-recipe skill includes a
[two-backend base](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-base.yaml)
and [candidate](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-candidate.yaml).
The candidate keeps both provider definitions and the listener, adds a `code`
recipe, and publishes `example/code` while preserving `vllm-sr/auto` as the
default. The endpoints are placeholders; reachability and answer quality
require real backends.

From a source checkout with the v0.5 CLI, record the commit and fixture
digests, then validate both files and inspect their exact diff:

```bash
git rev-parse HEAD
shasum -a 256 tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-*.yaml
vllm-sr config validate --config tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-base.yaml
vllm-sr config validate --config tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-candidate.yaml
diff -u tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-base.yaml tools/agent/skills/vllm-sr-configure-recipe/fixtures/two-backend-candidate.yaml
```

The fixture test round-trips both documents through the canonical parser and
asserts that the candidate changes only recipes and entrypoints. Offline CLI
validation may report that only structural checks ran. On a running stack,
also use `config plan --mode replace` against its management endpoint, confirm
the active config identity, and preview `example/code` with a bounded request.
A proposal has not changed the stack; apply and routed delivery are separate
reviewed steps.

## Example 2: paired evaluation

The benchmark skill's
[two-case source](https://github.com/vllm-project/semantic-router/blob/main/tools/agent/skills/vllm-sr-benchmark/fixtures/two-case-mcq.jsonl)
contains synthetic public questions. The local preparation command in its
[smoke walkthrough](https://vllm-sr.ai/install/agent/vllm-sr-benchmark/references/two-case-smoke.md)
produces a `smoke` development manifest with two cases and a source digest;
it makes no model calls. The fixture test verifies the case count, source
digest, split and normalized answers.

For an actual A/B result, freeze two live manifests over the same case IDs,
grader, request profile and prices; bind each Router target to its verified
active config hash. Review the planned calls and cost before dispatch. Save
both run IDs, use sr-bench's `comparison-options` and `compare`, and report
paired/failed/missing denominators with the conservative quality interval.
The [comparison example](https://vllm-sr.ai/install/agent/vllm-sr-compare-configurations/references/paired-evidence.md)
shows why a valid two-case calculation still gives insufficient evidence for
broad quality or savings claims. It contains illustrative values, not a live
project result.

## Example 3: decision-model and routing failure

Use the [decision-model skill](https://vllm-sr.ai/install/agent/vllm-sr-decision-model/SKILL.md)
to resolve a recipe's question to a deployment and check its model card and
readiness. On the actual public entrypoint, capture a traced route preview
and a bounded route probe through the listener. The
[diagnosis walkthrough](https://vllm-sr.ai/install/agent/vllm-sr-diagnose-routing/SKILL.md#three-case-diagnostic-walkthrough)
separates an expected route, an unknown/fallback, and an upstream delivery
failure. Preserve the request digest, active config version, `signal_errors`,
`eval_trace`, selected decision/model, response path, headers and delivery
assertion. An HTTP 200 or `/health` response alone does not settle which
boundary worked.

## What is verified here

The authored/public mirror, catalog and command checks protect these links and
CLI examples. The two-backend fixture passes the canonical parser, and the
synthetic dataset prepares deterministically without generation. Live route,
model-runtime and paired benchmark transcripts require reachable backends and
an sr-bench worker. Until those run IDs and receipts are attached, these
examples demonstrate the workflow and its stopping points, not a measured
performance result. Keep credentials, private prompts and hostnames out of
published evidence.

### Offline fixture receipt

At source commit `da9d5bf0fcabaf9c00f08981e3d117ee14902c96`, the
checked-out CLI reported `vllm-sr version: 0.5.0`. The canonical parser
accepted both two-backend documents, and the fixture test confirmed that only
recipe and entrypoint fields changed. Their SHA-256 digests were:

| File | SHA-256 |
| --- | --- |
| `two-backend-base.yaml` | `8bf56c9c2f9a567eb84ada0d463259336bb99de5f13cf2ae2306be6cc690fdca` |
| `two-backend-candidate.yaml` | `6752790703e4b640dd054a601b2d89bd1ed97b8c423935cb0aeb09639cb48333` |
| `two-case-mcq.jsonl` | `b8a1eee4958fd029a8dab87e88ba1a1e9d22877bf90701ec688475c9c47b2ffc` |

The local sr-bench dataset preparation returned `case_count: 2`,
`profile: smoke`, `split: dev`, `custom_subset: true`, and dataset SHA-256
`0da687fda6ca748a07a0b0acc555f75f5f8d61976e418adb30391688e054fe20`.
Those checks made no generation calls. Offline `config validate` reported that
only structural checks ran because no Router image or management endpoint was
available. No `config plan`, route preview/probe, replay or live comparison
result is claimed by this receipt.
