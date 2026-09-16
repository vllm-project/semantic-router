# Operator API contract

- API types declare the CRD schema; webhooks own admission semantics;
  controllers own translation and reconciliation.
- Update generated CRDs, sample fixtures, and webhook regression tests with an
  API contract change.
- Do not put controller-side canonical config translation into API types or
  webhook files.
