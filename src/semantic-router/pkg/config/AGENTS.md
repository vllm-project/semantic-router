# Config Package Notes

## Scope

- `src/semantic-router/pkg/config/**`
- local rules for config schema hotspots, especially `config.go`

## Responsibilities

- Keep `config.go` focused on the central schema table and shared config contracts.
- Treat adjacent files as the place for plugin contracts, validators, and load/registry helpers.
- Keep layer-specific contracts distinct:
  - signal definitions own extraction-oriented config
  - decision config owns boolean composition and control logic
  - algorithm config owns per-decision model-selection policy
  - plugin config owns post-decision processing behavior
  - global config owns intentionally cross-cutting behavior

## Change Rules

- Do not add new plugin structs, helper decoders, or utility walkers back into `config.go` if they can live in an adjacent file.
- Do not collapse signal, decision, algorithm, plugin, and global config into one catch-all struct when separate contracts or support files can keep ownership clear.
- Preferred split:
  - core schema tables in `config.go`
  - signal / decision / algorithm / plugin contracts in dedicated support files where the schema is large enough to justify separation
  - plugin contracts/helpers in dedicated `*_plugin.go` or support files
  - validation in `validator.go`
  - load/registry behavior in `loader.go` and `registry.go`
- If a change touches `config.go`, prefer a net reduction in file size or responsibility count.
