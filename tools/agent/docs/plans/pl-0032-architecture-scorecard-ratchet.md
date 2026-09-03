# PL-0032: Architecture Debt Consolidation

## Goal

Keep one current execution owner for non-release architecture debt and retire
exceptions as the corresponding code becomes modular and directly testable.

## Scope

This plan owns every open entry indexed by
[`../tech-debt/README.md`](../tech-debt/README.md):

- repository structure-rule exceptions;
- fleet-sim lint, structure, and optimizer boundaries;
- classification subsystem boundaries;
- native binding structure and complexity exceptions;
- Router Flow state-store validation;
- backend-target producer and consumer parity;
- reviewed content moderation;
- ONNX binding runtime coverage.

Release planning and daily GitHub queue state are out of scope.

## Exit Criteria

- Every open debt entry has this plan as its owner until it is promoted into an
  active release plan.
- Each entry is retired from current docs when its source-level exit criteria
  pass.
- Structure, lint, and runtime gates report new regressions without hiding them
  behind broad exceptions.

## Task List

- [ ] `ADC-01` Reduce or narrow the remaining structure-rule exceptions.
- [ ] `ADC-02` Align fleet-sim lint and structure checks with the shared
  repository contract.
- [ ] `ADC-03` Split classifier construction, discovery, and request-time
  orchestration into narrower owner modules.
- [ ] `ADC-04` Separate fleet-sim analytical sizing, simulation verification,
  reporting, and public exports.
- [ ] `ADC-05` Validate Router Flow's Redis state store and document deployment
  guidance.
- [ ] `ADC-06` Replace hidden moderation behavior with reviewed policy and code.
- [ ] `ADC-07` Add mandatory CPU-compatible ONNX binding coverage.
- [ ] `ADC-08` Converge backend-target validation and preservation across
  canonical configuration producers and consumers.

## Next Action

Choose the highest-impact open debt entry, verify its current source evidence,
and either narrow its scope or implement one independently reviewable exit
criterion.

## Operating Rules

- Keep one unresolved gap per debt entry.
- Use current code and gate output as evidence; do not copy pull-request
  narratives into debt files.
- Remove closed entries and completed tasks from the active tree.

## Related Docs

- [Technical Debt](../tech-debt/README.md)
- [Architecture Guardrails](../architecture-guardrails.md)
- [Architecture Status](../architecture-scorecard.md)
