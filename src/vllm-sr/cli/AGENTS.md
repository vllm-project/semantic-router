# vLLM-SR CLI

- `core.py` and `commands/runtime.py` own the user-facing local runtime flow;
  image selection, container wiring, readiness, and platform support remain
  independently testable helpers.
- `docker_cli.py` is a compatibility re-export seam, not a runtime owner.
- Keep config parsing/migration/validation out of runtime orchestration.
- Canonical field inventories are shared; do not duplicate them across
  `models.py`, `config_migration.py`, and `validator.py`.
- Legacy config handling remains an explicit migration path.
