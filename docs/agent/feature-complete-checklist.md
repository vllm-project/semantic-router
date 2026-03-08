# Feature Complete Checklist

A feature is not done until all applicable checks below are satisfied.

## Required

- Lint passes for all touched languages
- Structure gates pass for changed files
- Relevant fast tests pass
- Relevant feature/integration tests pass
- Local image startup smoke passes for non-doc code changes
- Affected local E2E profiles pass
- CI covers the remaining affected profiles

## E2E Expectation

- If user-visible behavior changes, update or add at least one E2E case
- Pure refactors may skip new E2E coverage only when behavior is unchanged

## Standard Report Format

- Primary skill: name and why it was selected
- Impacted surfaces: required surfaces plus conditional surfaces that were actually hit
- Conditional surfaces intentionally skipped: name each skipped surface and why
- Scope: what changed
- Environment: `cpu-local` or `amd-local`
- Fast gate: commands and results
- Feature gate: commands and results
- Local smoke: container, status, dashboard, router
- Local E2E: profiles and results
- Follow-up: risks, skipped checks, external blockers, and any durable tech-debt item added or updated because the code still diverges from the desired architecture
