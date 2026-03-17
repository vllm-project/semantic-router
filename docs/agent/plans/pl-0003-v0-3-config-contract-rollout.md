# v0.3 Config Contract Rollout

## Goal

Finish the repo-wide rollout of the v0.3 canonical config contract where Go owns the schema, canonical YAML uses `version/listeners/providers/routing/global`, and the DSL owns only the routing surface.

## Scope

- Go config parser/normalizer/export helpers
- DSL compile/decompile/WASM contract
- dashboard backend/frontend config flows
- operator and Helm config generation
- remaining legacy CLI migration work
- tests and harness validation aligned to the new contract

## Exit Criteria

- CLI, dashboard, operator, and Helm all read or write the same canonical config shape for the common path.
- DSL compile/decompile only owns `routing.modelCards`, `routing.signals`, and `routing.decisions`.
- legacy `providers.models[].endpoints/access_key` and `router-defaults.yaml` are retired from steady-state workflows.
- dashboard frontend no longer exposes backend/global DSL editing paths as if they were deployable routing state.
- harness validation and E2E coverage reflect the new contract instead of the removed legacy surfaces.

## Task List

- [x] P01 Add canonical Go parser/defaults and router-owned global defaults.
- [x] P02 Stop Python CLI serve path from generating `.vllm-sr/router-config.yaml`.
- [x] P03 Switch dashboard backend defaults/update paths to canonical `global` overrides.
- [x] P04 Switch operator-generated config output to canonical top-level layout.
- [x] P05 Add routing-only DSL YAML emission and routing-only decompile path.
- [x] P06 Add top-level DSL model catalog support for routing-owned semantic model metadata.
- [x] P07 Remove or replace dashboard/frontend BACKEND and GLOBAL DSL authoring flows.
- [x] P08 Migrate CRD and Helm value/schema surfaces to canonical `providers/routing/global`.
- [x] P09 Add explicit `config migrate` coverage for old nested provider endpoint/auth layouts.
- [x] P10 Retire remaining legacy DSL compile/decompile expectations and related docs/tests.
- [x] P11 Run full harness gates and close remaining structural hotspot debt exposed by this rollout.

## Current Loop

- Current focus: rollout closed for the steady-state v0.3 contract; keep future follow-up limited to normal feature work and unrelated structural debt.
- Last completed loop: removed the router's steady-state legacy runtime parser fallback, migrated the remaining maintained deploy/E2E/example assets to canonical v0.3, added maintained-asset contract tests, regenerated CRDs, and retired TD001.
- Next loop: no rollout-specific work remains; future changes should land against the canonical contract and open new debt only if they introduce a fresh architectural gap.

## Decision Log

- 2026-03-14: Keep `default_model`, `reasoning_families`, and `default_reasoning_effort` in `providers`, not DSL.
- 2026-03-14: Keep semantic catalog data in `routing.modelCards`, and place deployment bindings in `providers.models[].backend_refs[]`.
- 2026-03-14: Keep onboarding remote import compatible with full canonical YAML, while DSL import of the same YAML only decompiles the `routing` surface.
- 2026-03-14: Retire repo-owned full example config trees and replace them with canonical `config/config.yaml` plus `config/signal|decision|algorithm|plugin` fragments.
- 2026-03-14: Remove legacy DSL `BACKEND`/`GLOBAL` authoring support completely instead of keeping it as a migration-era compatibility surface.
- 2026-03-14: Remove CLI-side `router-defaults.yaml` and merger helpers; router-owned defaults plus explicit `config migrate` are the only supported path.
- 2026-03-14: Treat `routing.models`, `providers.model_targets`, `providers.backends`, and `providers.auth_profiles` as removed contract fields; parser paths now fail fast instead of silently accepting them.
- 2026-03-14: Treat router-owned capability modules as part of `global.model_catalog`; module settings now live under `global.model_catalog.modules`, not as a peer top-level global block.
- 2026-03-15: Normalize dashboard config editing around canonical `providers/routing/global` on both read and save, so editing a legacy file no longer writes `model_config`/`vllm_endpoints` back to disk.
- 2026-03-15: The steady-state router parser now rejects deprecated legacy user config directly; the remaining TD001 scope is maintained harness/deploy/example assets and internal adapters that still need migration.
- 2026-03-16: The final maintained deploy/E2E/example assets and embedded `values.yaml` config blocks were migrated to canonical v0.3, maintained-asset contract tests were added, and TD001 was retired.

## Follow-up Debt / ADR Links

- [TD006 Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots](../tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md)
