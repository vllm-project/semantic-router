# Contract generators

This directory owns build-time generators and templates for checked-in public
contracts:

- `configschema/`: Router JSON Schema and dashboard TypeScript contracts;
- `openapi/`: the Router API specification;
- `crd/`: the operator CRD reference;
- `embed_generated_index.py`: the API endpoint index embedded in documentation;
- `boilerplate.go.txt`: the shared Go header used by Kubernetes generators.

Run generators through their repository Make targets so the module working
directory, native dependencies, and generated outputs stay consistent:

```bash
make generated-contract-generate
make generated-contract-check
make docs-crd
make docs-crd-check
```

Runtime schema and API implementations remain in their owning source packages.
