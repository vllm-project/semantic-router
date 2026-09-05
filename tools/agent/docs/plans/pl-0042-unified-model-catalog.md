# PL-0042: Unified Model Catalog and Evaluation Index

## Goal

Deliver one versioned catalog and materialization path for provider, protocol,
model, provider mapping, reasoning, presentation, evaluation, and index metadata so
Day-0 support updates every product surface from one source.

## Scope

- Review and settle the public catalog/config contract.
- Implement schemas, compiler, provenance, and immutable effective registry.
- Replace duplicated Router, CLI, Dashboard, and website inventories.
- Replace scalar quality metadata with evaluation records and versioned indices.
- Generate the Dashboard Add Model experience and public Models/leaderboard
  views from the catalog.
- Establish a broad current-and-previous-generation physical-model baseline;
  keep GPT-6 Astra for the separate representative Day-0 change.
- Validate the contribution flow with representative model and provider work.

## Non-Goals

- Runtime discovery of arbitrary internet models.
- A compatibility claim for every Dashboard provider preset.
- Redistribution of benchmark data without permission.
- Combining intelligence, efficiency, cost, and availability into one opaque
  score.
- A new top-level configuration hierarchy or catalog build metadata in user
  YAML.

## Exit Criteria

- One validated source graph feeds every runtime and presentation consumer.
- Built-in and handwritten model cards materialize through the same typed path.
- Provider/API/reasoning defaults no longer depend on central config-helper
  switches or parallel Dashboard data.
- The default intelligence index exposes components, version, coverage,
  status, and provenance, and missing data never becomes zero.
- The Dashboard and website render generated provider/model support data,
  presentation assets, and evidence-backed leaderboards.
- A model-only Day-0 change is complete through catalog data, conformance,
  generated surfaces, documentation, and selected CI gates.

## Task List

- [x] `TASK-01` Publish the unified catalog, config, scoring, UX, website, and
  Day-0 workflow proposal with the current split inventory.
- [x] `TASK-02` Add versioned resource schemas, source layout, compiler,
  provenance, immutable registry, fixtures, and generated-diff enforcement.
- [x] `TASK-03` Add the v0.3 catalog materializer and targeted migration tool
  for aliases, backend Provider IDs, built-in overlays, and custom model cards.
- [x] `TASK-04` Move protocol, auth, provider, provider mapping, and reasoning behavior
  to catalog-backed registries and remove config-helper glue.
- [x] `TASK-05` Add evaluation records, the default intelligence index, score
  resolver, missing-data policy, and runtime observed-quality separation.
- [x] `TASK-06` Move Dashboard Add Model and provider logos to the catalog API;
  generate the public Models support matrix and leaderboards.
- [ ] `TASK-07` Complete config, protocol, UI, website, and affected E2E
  validation. The representative new-model Day-0 change lands separately after
  this architecture PR.
- [ ] `TASK-08` Remove superseded inventories and retire TD058 after every exit
  criterion is enforced.

## Next Action

Format and validate the complete v0.3 vertical slice and broad baseline catalog
on the AMD validation host, exercise a real catalog-backed model through the
Router and Dashboard, then close the plan and TD058. GPT-6 Astra remains the
separate focused Day-0 follow-up.

## Operating Rules

- Keep catalog facts, user bindings, runtime observations, and presentation
  views separate.
- Keep missing, failed, unavailable, not-applicable, and zero distinct.
- Add code adapters only for real wire-semantic differences.
- Preserve benchmark license, provenance, and redistribution boundaries.
- Run the smallest harness-selected gate first and expand only after it passes.
- Keep implementation modules narrow and extract from existing hotspots before
  adding responsibility.

## Related Docs

- [Unified Model Catalog and Evaluation Index](../../../../website/docs/proposals/unified-model-catalog-and-evaluation-index.md)
- [TD058: Fragmented Model Support Metadata](../tech-debt/td-058-fragmented-model-support-metadata.md)
- [Evaluation Plane](../../../../website/docs/benchmarking/evaluation-plane.md)
- [Dashboard Modeling Experience](pl-0038-dashboard-modeling-experience.md)
- [Evaluation Plane Plan](pl-0039-evaluation-plane.md)
- [Architecture guardrails](../architecture-guardrails.md)
- [Feature-complete checklist](../feature-complete-checklist.md)
