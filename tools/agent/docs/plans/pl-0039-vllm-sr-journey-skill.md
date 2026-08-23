# PL-0039: vLLM-SR Contributor Journey Skill

## Goal

Deliver the contributor-facing agent skill and helper tooling for deployment,
recipe generation, contract validation, evaluation, reviewed tuning, and explicit
activation described in issue #2977.

## Scope

- Contributor support skill and registry wiring
- `vllm_sr_journey.py` orchestration helper
- `vllm-sr recipe scaffold` CLI command
- Illustrative example under `config/recipes/examples/journey-starter/`
- Public tutorial and CONTRIBUTING pointer

## Non-Goals

- Autonomous production activation or scheduled tuning pipelines
- New unsupported deployment targets
- In-router LLM config generation services

## Exit Criteria

- [x] `vllm-sr-journey` skill is registered and validated by `make agent-validate`
- [x] Journey helper supports detect-env, validate, evaluate, and review
- [x] `vllm-sr recipe scaffold` emits validated five-file recipes
- [x] Example journey and website tutorial are indexed
- [x] Applicable harness gates pass for changed surfaces

## Task List

- [x] `JOURNEY-01` Skill charter and registry wiring
- [x] `JOURNEY-02` `vllm_sr_journey.py` helper and make target
- [x] `JOURNEY-03` `vllm-sr recipe scaffold` command and tests
- [x] `JOURNEY-04` Example journey and website tutorial
- [x] `JOURNEY-05` Harness validation and feature gates

## Next Action

None. Close PL-0039 after merge unless follow-up work extends the contributor journey.

## Operating Rules

- Generation is not acceptance; review bundles stay `activated: false` until a
  human approves activation.
- Private environment details stay out of committed artifacts.
- Examples under `config/recipes/examples/` are illustrative and excluded from
  maintained catalog discovery.

## Related Docs

- Issue [#2977](https://github.com/vllm-project/semantic-router/issues/2977)
- [`tools/agent/skills/contributor/vllm-sr-journey/SKILL.md`](../../skills/contributor/vllm-sr-journey/SKILL.md)
- [`website/docs/tutorials/agent/vllm-sr-journey.md`](../../../../website/docs/tutorials/agent/vllm-sr-journey.md)
- [`config/recipes/CONFORMANCE.md`](../../../../config/recipes/CONFORMANCE.md)
