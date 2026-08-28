# PL-0038: MoM First-Class Evaluation Epic

## Goal

Implement reproducible end-to-end evaluation for every published
Mixture-of-Models (MoM) under issue #3038, including core suite enforcement,
extension packs, regression gates, and model-card scorecard publication for all
five MoM V1 entrypoints.

## Scope

- Versioned MoM evaluation manifest and result schema
- Mandatory core suite v1 and standalone baseline protocol
- Extension pack registry (cost, latency, security, orchestration)
- Unified runner under `bench/mom_eval/`
- Scorecard publication and historical retention
- Reference smoke scorecards for MoM V1 1.0.0

## Non-Goals

- Replacing decision-level routing evaluation (#2333)
- Evaluating Router Models in isolation
- Hosted benchmark SLA or always-on formal runs in CI without backends

## Exit Criteria

- [x] `MOM-01` Contract schemas, core suite, baseline protocol, pack registry
- [x] `MOM-02` Validation tooling and recipe conformance integration
- [x] `MOM-03` Runner, collectors, regression, failure slicing, publish pipeline
- [x] `MOM-04` Make targets and smoke CI workflow
- [x] `MOM-05` Reference scorecards for all five MoM V1 entrypoints
- [x] `MOM-06` Documentation and maintainer skill

## Next Action

Run formal evaluations on maintainer hardware with live backends and replace
smoke synthetic metrics with publishable formal results.

## Related Docs

- [MoM evaluation runner guide](../../../../bench/mom_eval/README.md)
- [MoM evaluation user guide](../../../../website/docs/benchmarking/mom-evaluation.md)
- [Issue #3038](https://github.com/vllm-project/semantic-router/issues/3038)
