---
name: routing-calibration-loop
category: support
description: Calibrates immutable routing manifests with executable probes, offline DSL validation, controlled process replacement, and structured failure review. Use when tuning signals, projections, decisions, or maintained route examples against a real apiserver.
---

# Routing Calibration Loop

## Trigger

- Use when a signal, projection, decision, or maintained routing example needs to be checked against a live router apiserver
- Use when a routing failure must be classified as a bad probe, bad routing policy, or bad validator rule instead of blindly patching the profile
- Use when a maintainer wants the loop `eval -> update -> validate -> replace -> eval` to be run with versioned source evidence

## Required Surfaces

- `harness_docs`

## Conditional Surfaces

- `harness_exec`
- `router_service_platform`
- `router_config_contract`
- `signal_runtime`
- `decision_logic`
- `algorithm_selection`
- `dsl_config_contract`
- `docs_examples`

## Stop Conditions

- No live router base URL is available and no local replacement environment has been chosen
- No probe manifest exists and the task cannot safely infer executable probes from maintained examples
- A replacement would change remote runtime state without preserving the current deployment input or rollback path
- Local validation fails for reasons that are not yet understood or recorded

## Workflow

1. Start from executable probes, not prose examples.
   - Prefer a machine-readable manifest. [`config/recipes/balance/probes.yaml`](../../../../../config/recipes/balance/probes.yaml) is the default maintained example, not the only supported target.
   - The manifest should stay profile-generic: point to any owned routing YAML / DSL pair through `routing_assets`, and group probes by decision with multiple variants when robustness matters.
   - Treat each probe as both a test case and a specification fragment.
2. Baseline the live router before editing policy.
   - Use [`tools/agent/scripts/router_calibration_loop.py`](../../../../../tools/agent/scripts/router_calibration_loop.py) to run `/api/v1/eval` across the probe suite and preserve the report beside the source manifest.
   - Record which decision actually fired, which signals matched, and which signals were expected but absent.
3. Classify every failure under one of three buckets before changing anything.
   - `query_quality`: the prompt is not a robust representative of the intended route.
   - `routing_design`: the signal / projection / decision design is too broad, too narrow, or too brittle.
   - `validator_quality`: the runtime behavior is reasonable but static validation is over-reporting or under-reporting.
4. Edit the canonical authoring surface locally.
   - For maintained routing, edit the owned YAML / DSL asset pair instead of patching only the live server.
   - Do not add narrow trigger-phrase hacks just to pass one probe.
5. Run offline validation before replacing the deployment.
   - Use the runner's `validate` path to execute `sr-dsl validate` against the DSL source, or against a YAML file through decompile-then-validate.
   - Prefer manifest-owned assets as defaults, but allow explicit YAML / DSL overrides for any other routing profile.
   - Keep validation output with the loop artifacts so validator behavior can be reviewed alongside runtime eval output.
6. Replace the immutable deployment and re-evaluate.
   - Promote the validated manifest through the owning Docker or Kubernetes deployment workflow; the standalone Router API is intentionally not a config writer.
   - Preserve the previous deployment input as the rollback artifact, then wait for `GET /ready` to return `ready=true` before trusting eval results.
   - Re-run the same probe suite after replacement and compare before / after success rate and per-probe traces.
7. Close the loop with structured reflection.
   - `0. Query quality`: Is the probe semantically representative, or is it a brittle phrase trigger?
   - `1. Routing design`: Are the signal, projection, and decision boundaries robust, or merely sufficient for this probe set?
   - `2. Validator quality`: Do warnings or failures reflect real ambiguity, or missing static semantics?
8. If a durable architecture gap remains, update the indexed debt entry instead of leaving the mismatch only in chat or the report.

## Gotchas

- A standalone router compiles one manifest at startup. Calibration must replace that complete deployment input rather than patching a live process.
- Do not declare success just because one crafted query passes. Probe quality is part of the task; decision-level robustness should be checked with multiple variants, not just one trigger phrase.
- If runtime eval looks correct and validation still looks wrong, assume validator semantics may need work rather than forcing a worse route design.
- If replacement succeeds but success rate regresses, restore the preserved deployment input before continuing.

## Must Read

- [AGENTS.md](../../../../../AGENTS.md)
- [website/docs/installation/amd-rocm.md](../../../../../website/docs/installation/amd-rocm.md)
- [config/recipes/balance/probes.yaml](../../../../../config/recipes/balance/probes.yaml)
- [tools/agent/scripts/router_calibration_loop.py](../../../../../tools/agent/scripts/router_calibration_loop.py)

## Standard Commands

- `python3 tools/agent/scripts/router_calibration_loop.py eval --router-url http://<router-host>:8080 --probes <profile>.probes.yaml`
- `python3 tools/agent/scripts/router_calibration_loop.py validate --yaml <routing>.yaml --dsl <routing>.dsl`
- `make agent-report ENV=amd CHANGED_FILES="config/recipes/balance/config.yaml,config/recipes/balance/recipe.dsl,website/docs/installation/amd-rocm.md"`
- `make agent-ci-gate CHANGED_FILES="tools/agent/skills/maintainer/routing-calibration/SKILL.md,tools/agent/scripts/router_calibration_loop.py,config/recipes/balance/probes.yaml"`

## Acceptance

- Each calibration round preserves before / after probe reports, live decision traces, and the exact versioned deployment inputs
- Failures are explicitly reviewed under query quality, routing design, and validator quality instead of being patched blindly
- Maintained routing changes are validated offline before process replacement and re-evaluated on the live endpoint afterward
- The loop leaves behind executable probes or maintained examples that are stronger than the ones it started with, ideally by improving decision-level variant coverage instead of adding single-example hacks
