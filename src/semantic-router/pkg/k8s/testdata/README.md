# Kubernetes converter fixtures

These fixtures test the conversion from `IntelligentPool` and
`IntelligentRoute` resources to the router's canonical v0.3 configuration.
They are test inputs and golden outputs, not deployment examples.

## Layout

```text
testdata/
├── base-config.yaml  # Router-wide fields that do not come from the CRDs
├── input/            # One IntelligentPool + IntelligentRoute scenario per file
└── output/           # Expected canonical YAML for the matching input file
```

`base-config.yaml` supplies shared provider defaults, services, stores,
integrations, and model assets. The converter supplies the pool models,
provider bindings, routing signals, decisions, and route-local plugins.

## Scenarios

The numbered pairs cover:

- a minimal pool and route (`01`);
- keyword, embedding, and domain signals alone and in combination (`02`–`08`);
- those signal families with route-local plugins (`09`–`15`);
- a multi-decision route without plugins (`16`); and
- multimodal embedding conversion (`17`).

Keep an input and its output under the same filename. Add a new numbered pair
when a converter behavior needs independent coverage; update an existing pair
only when its contract intentionally changes.

The output files contain the complete canonical configuration after merging
the CRD-derived fields with `base-config.yaml`. They should use current
`providers`, `routing`, and `global` fields rather than legacy config aliases.

## Run the fixture test

From the router module:

```bash
cd src/semantic-router
go test ./pkg/k8s -run TestConverterWithTestData
```

The test converts every input, compares it with the corresponding golden
output, and parses the result through the runtime config loader. Review golden
file changes as API changes: a passing diff still needs to represent the
intended Kubernetes-to-router contract.
