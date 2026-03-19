# vllm-sr CLI Notes

## Scope

- `src/vllm-sr/cli/**`
- local rules for CLI orchestration and the `docker_cli.py` hotspot

## Responsibilities

- Keep CLI files centered on one dominant command-orchestration responsibility.
- Treat `docker_cli.py` as the ratcheted hotspot for top-level command flow and user-facing runtime decisions.
- Treat `models.py`, `config_migration.py`, and `validator.py` as config-contract hotspots whose ownership should stay split between schema, compatibility, and semantic validation.

## Change Rules

- Move docker image resolution, container wiring, readiness helpers, and platform-specific support into adjacent modules instead of growing `docker_cli.py`.
- Prefer extraction-first edits when adding new serve/start/status behavior.
- Keep parser/schema/merger logic out of runtime orchestration files unless the change truly belongs to config translation.
- Do not encode the same canonical field inventory or signal/plugin type list independently across `models.py`, `config_migration.py`, and `validator.py` when one shared inventory or helper can own it.
- When adding new config-contract surface area, prefer dedicated schema-family or validation helpers instead of growing the all-in-one `models.py` / `validator.py` hotspots.
- Keep legacy migration behavior behind explicit compatibility helpers; do not turn `config_migration.py` into a second general-purpose config runtime.
