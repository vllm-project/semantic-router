# Dashboard HTTP handlers

- Handlers own HTTP method checks, decoding, response encoding, and delegation.
- Keep canonical config persistence/rollback/runtime apply separate from status
  collection and container or supervisor probing.
- `config.go` and `deploy.go` orchestrate config endpoints; status collection
  belongs in the existing status collectors rather than those files.
- Do not duplicate router config schema or mutation rules in transport code.
