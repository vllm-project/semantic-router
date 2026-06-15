# Dashboard Pages Notes

## Scope

- `dashboard/frontend/src/pages/**`
- local rules for dashboard route-level pages and large-page hotspots

## Responsibilities

- Page files should own page orchestration, route-level state, and section composition.
- Keep page-local schema/types, constant tables, and pure helper functions in adjacent support files when they start crowding the page.

## Change Rules

- `BuilderPage.tsx` is a ratcheted hotspot. Keep route state, store wiring, and editor-mode orchestration in the page, but move toolbars, overlays, output panels, builders, and support hooks into sibling modules.
- `DashboardPage.tsx` is an overview hotspot. Keep page-level fetch cadence and navigation orchestration there, but move stat-card shaping, flow-diagram support, and large section renderers into sibling modules.
- `ConfigPage.tsx` is a legacy hotspot. New config schema helpers or duplicated shape definitions should move into adjacent support modules instead of growing the page file.
- `ConfigPageSignalsSection.tsx` and `ConfigPageModelsSection.tsx` are config-form hotspots. Keep section orchestration there, but move canonical/legacy translation helpers, field-shape tables, and repetitive mapper logic into sibling support modules.
- `SetupWizardPage.tsx` is also a ratcheted hotspot. Keep async validation and activation orchestration in the page, but move wizard step panels, config builders, and provider constants into sibling modules.
- `OpenClawPage.tsx` is a route-level hotspot. Keep feature composition and route state there, but move graph shaping, fallback-node assembly, status polling helpers, and modal support into sibling modules.
- `MLSetupPage.tsx` is a workflow hotspot. Keep page-level setup orchestration there, but move provider-specific step definitions and large support helpers into adjacent files.
- Prefer extracting repeated section renderers or config data helpers before adding more inline page logic.
