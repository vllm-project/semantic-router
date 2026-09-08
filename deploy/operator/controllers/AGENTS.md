# Operator controllers

- Keep reconciliation, backend discovery, platform integration, and
  operator-to-router config translation as distinct concerns.
- `canonical_config_builder.go` owns translation, not CRD schema or admission
  validation.
- Keep OpenShift/gateway discovery outside config-family translation so either
  side can be tested independently.
