# Routing core contract

- This package and its subpackages import no Envoy types and no `pkg/extproc`
  (`tools/agent/structure-rules.yaml` enforces it). Adapters depend on it.
- `apply.go` reproduces how Envoy applies ext_proc mutations and renders local
  replies. Change it only against Envoy's source for the shipped version, and
  keep `pkg/extproc`'s parity goldens unchanged.
- `parity` normalizes volatile values only. Do not widen the normalizer to hide
  a real difference between gateway modes; fix the difference or document it in
  the design doc.
