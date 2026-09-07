# Router configuration contract

- Steady-state input is canonical `version/listeners/providers/routing/global`.
  Legacy layouts belong only in migration tooling.
- `providers.defaults` owns default selection and reasoning metadata;
  `providers.models[]` owns concrete backend bindings.
- `global` remains layered into `router`, `services`, `stores`, `integrations`,
  and `model_catalog`.
- Signals extract facts; decisions compose them; algorithms select models;
  plugins process the selected route. Keep their schemas and validators owned
  by those families.
- Update the router surface catalog, `config/` fragments, schemas/generated
  assets, config tests, and current public docs when their shared contract
  changes.
- Canonical import/export and normalization stay in `canonical_*.go`; runtime
  loading must not silently perform migration.
