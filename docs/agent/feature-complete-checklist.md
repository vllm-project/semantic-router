# Feature Complete Checklist

A feature is not done until all applicable checks below are satisfied.

## Required

- Lint passes for all touched languages
- Structure gates pass for changed files
- Relevant fast tests pass
- Relevant feature/integration tests pass
- Local image startup smoke passes for non-doc code changes
- Behavior-sensitive E2E coverage uses explicit acceptance thresholds instead of report-only metrics
- CI covers the remaining affected profiles
- Manual local E2E or workflow-driven integration runs are recorded when they were needed for debugging or risk reduction

## Loop Completion

- A task is not done when a failing gate is merely observed.
- The active loop continues until the applicable gates pass for the current change or subtask, or an external blocker is explicitly reported.
- Long-horizon multi-subtask work keeps its execution plan current as part of the done state.

## E2E Expectation

- If user-visible behavior changes, update or add at least one E2E case
- Report-only or `0%`-only tests do not count as behavior coverage
- Benchmark and stress probes may supplement E2E coverage, but they do not replace acceptance tests unless they declare explicit pass/fail budgets
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
- Manual local E2E: profiles and results, if run
- Follow-up: risks, skipped checks, external blockers, and any durable tech-debt item added or updated because the code still diverges from the desired architecture
