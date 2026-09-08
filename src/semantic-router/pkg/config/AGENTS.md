# Router configuration contract

- Steady-state input is canonical `version/listeners/providers/routing/global`.
  Legacy layouts belong only in migration tooling.
- `providers.defaults` owns default model and reasoning-effort selection;
  `providers.models[]` owns aliases, optional catalog references, custom
  reasoning metadata, and concrete backend bindings. The repository catalog
  owns built-in reasoning-family and provider/protocol metadata.
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
