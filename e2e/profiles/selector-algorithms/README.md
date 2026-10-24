# Selector algorithms E2E profile

This profile hosts deterministic deployed-path contracts for algorithms whose
runtime catalog execution tier is `selector`. Looper coverage remains in the
`looper` profile.

The first contract covers `static`. A keyword fixture selects a decision with
two ordered candidate models, then repeated requests prove that the runtime:

- matched the intended decision;
- executed the explicitly configured `static` selector;
- consistently selected the first candidate.

Run the profile from the repository root:

```bash
make e2e-test E2E_PROFILE=selector-algorithms
```

Future algorithm PRs should add their deterministic fixtures and testcase to
this profile, then update
`e2e/pkg/testcases/testdata/selector_algorithm_coverage.json`.
