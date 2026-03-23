---
name: routing-calibration-loop
category: support
description: Calibrates routing changes against a live router endpoint with executable probes, local DSL validation, versioned deploys, and structured failure review. Use when tuning signals, projections, decisions, or maintained route examples against a real apiserver.
---

# Routing Calibration Loop

## Trigger

- Use when a signal, projection, decision, or maintained routing example needs to be checked against a live router apiserver
- Use when a routing failure must be classified as a bad probe, bad routing policy, or bad validator rule instead of blindly patching the profile
- Use when a maintainer wants the loop `eval -> update -> validate -> deploy -> eval` to be run with versioned evidence

## Required Surfaces

- `harness_docs`

## Conditional Surfaces

- `harness_exec`
- `router_service_platform`
- `router_config_contract`
- `signal_runtime`
- `decision_logic`
- `algorithm_selection`
- `dsl_crd`
- `docs_examples`

## Stop Conditions

- No live router base URL is available and no local replacement environment has been chosen
- No probe manifest exists and the task cannot safely infer executable probes from maintained examples
- A deploy would change remote runtime state without capturing the current version or without a rollback path
- Local validation fails for reasons that are not yet understood or recorded

## Workflow

1. Start from executable probes, not prose examples.
   - Prefer a machine-readable manifest. [`deploy/amd/balance.probes.yaml`](../../../../deploy/amd/balance.probes.yaml) is the default maintained example, not the only supported target.
   - The manifest should stay profile-generic: point to any owned routing YAML / DSL pair through `routing_assets`, and group probes by decision with multiple variants when robustness matters.
   - Treat each probe as both a test case and a specification fragment.
2. Baseline the live router before editing policy.
   - Use [`tools/agent/scripts/router_calibration_loop.py`](../../../../tools/agent/scripts/router_calibration_loop.py) to snapshot `/config/router` and `/config/versions`, then run `/api/v1/eval` across the probe suite.
   - Record which decision actually fired, which signals matched, and which signals were expected but absent.
3. Classify every failure under one of three buckets before changing anything.
   - `query_quality`: the prompt is not a robust representative of the intended route.
   - `routing_design`: the signal / projection / decision design is too broad, too narrow, or too brittle.
   - `validator_quality`: the runtime behavior is reasonable but static validation is over-reporting or under-reporting.
4. Edit the canonical authoring surface locally.
   - For maintained routing, edit the owned YAML / DSL asset pair instead of patching only the live server.
   - Do not add narrow trigger-phrase hacks just to pass one probe.
5. Run local validation before deploying.
   - Use the runner's `run` or `validate` path to execute `sr-dsl validate` against the DSL source, or against a YAML file through decompile-then-validate.
   - Prefer manifest-owned assets as defaults, but allow explicit YAML / DSL overrides for any other routing profile.
   - Keep validation output with the loop artifacts so validator behavior can be reviewed alongside runtime eval output.
6. Deploy durably and re-evaluate.
   - Use `/config/deploy` for versioned writes that persist and hot-reload.
   - After every deploy, wait for `GET /ready` to return `ready=true` before trusting `eval` results. Do not treat a successful deploy response as proof that router initialization has finished.
   - On current router builds, treat `GET /ready` as necessary but not always sufficient. Echo the live `/config/classification` payload back through `PUT /config/classification` as a runtime refresh barrier, then wait for `GET /ready` again before running probes.
   - Re-run the same probe suite after deploy and compare before / after success rate and per-probe traces.
7. Close the loop with structured reflection.
   - `0. Query quality`: Is the probe semantically representative, or is it a brittle phrase trigger?
   - `1. Routing design`: Are the signal, projection, and decision boundaries robust, or merely sufficient for this probe set?
   - `2. Validator quality`: Do warnings or failures reflect real ambiguity, or missing static semantics?
8. If a durable architecture gap remains, update the indexed debt entry instead of leaving the mismatch only in chat or the report.

## Gotchas

- Do not treat `PUT /config/classification` as the default deploy path for maintained routing work; it is useful for fast memory-only experiments but not for durable config changes.
- The runtime refresh barrier uses `PUT /config/classification`, but only after `/config/deploy` has already persisted the new config. The barrier is there to force classifier refresh, not to replace durable deploy.
- Do not declare success just because one crafted query passes. Probe quality is part of the task; decision-level robustness should be checked with multiple variants, not just one trigger phrase.
- If runtime eval looks correct and validation still looks wrong, assume validator semantics may need work rather than forcing a worse route design.
- If deploy succeeds but success rate regresses, capture the returned version and use the versions endpoint before continuing.

## Must Read

- [AGENTS.md](../../../../AGENTS.md)
- [docs/agent/README.md](../../../../docs/agent/README.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)
- [deploy/amd/README.md](../../../../deploy/amd/README.md)
- [deploy/amd/balance.probes.yaml](../../../../deploy/amd/balance.probes.yaml)
- owned routing YAML / DSL assets for the target profile, such as [deploy/recipes/balance.yaml](../../../../deploy/recipes/balance.yaml) and [deploy/recipes/balance.dsl](../../../../deploy/recipes/balance.dsl)
- [tools/agent/scripts/router_calibration_loop.py](../../../../tools/agent/scripts/router_calibration_loop.py)

## Standard Commands

- `python3 tools/agent/scripts/router_calibration_loop.py eval --router-url http://<router-host>:8080 --probes <profile>.probes.yaml`
- `python3 tools/agent/scripts/router_calibration_loop.py run --router-url http://<router-host>:8080 --probes <profile>.probes.yaml`
- `python3 tools/agent/scripts/router_calibration_loop.py run --router-url http://<router-host>:8080 --probes <profile>.probes.yaml --yaml <routing>.yaml --dsl <routing>.dsl`
- `python3 tools/agent/scripts/router_calibration_loop.py deploy --router-url http://<router-host>:8080 --yaml <routing>.yaml --dsl <routing>.dsl --ready-timeout 300`
- `make agent-report ENV=amd CHANGED_FILES="deploy/recipes/balance.yaml,deploy/recipes/balance.dsl,deploy/amd/README.md"`
- `make agent-ci-gate CHANGED_FILES="tools/agent/skills/routing-calibration-loop/SKILL.md,tools/agent/scripts/router_calibration_loop.py,deploy/amd/balance.probes.yaml"`

## Acceptance

- Each calibration round produces a probe report with before / after outcomes, live decision traces, and the captured deploy version when a deploy occurs
- Failures are explicitly reviewed under query quality, routing design, and validator quality instead of being patched blindly
- Maintained routing changes are validated locally before deploy and re-evaluated on the live endpoint after deploy
- The loop leaves behind executable probes or maintained examples that are stronger than the ones it started with, ideally by improving decision-level variant coverage instead of adding single-example hacks
