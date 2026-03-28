# E2E Testcase Notes

## Scope

- `e2e/testcases/**`

## Responsibilities

- Keep each testcase focused on one externally visible contract or one clearly named benchmark/reporting concern.
- Keep pass/fail assertions explicit and separate from metric collection or debug reporting.
- Keep shared helpers responsible for reusable execution and comparison mechanics, not for hiding acceptance semantics.
- Treat the baseline router contract as product behavior coverage, not as a place for silent report-only probes.

## Change Rules

- Do not use `0%`-only accuracy checks or "at least one request succeeded" as the acceptance bar for routing, classification, plugin, fallback, or API behavior tests.
- If a testcase is primarily a benchmark or stress-reporting probe, name and wire it as such; it does not replace acceptance coverage without explicit thresholds.
- When a testcase reports rates, accuracy, or latency, state the minimum acceptable floor in code or shared constants and keep the rationale easy to find.
- Prefer one contract per testcase file; if a file starts mixing unrelated routing, plugin, and transport assertions, split it before adding more cases.
