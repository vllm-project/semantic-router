# vllm-sr CLI Notes

## Scope

- `src/vllm-sr/cli/**`
- local rules for CLI runtime orchestration, compatibility barrels, and config-contract hotspots

## Responsibilities

- Keep CLI files centered on one dominant command-orchestration responsibility.
- Treat `core.py` and `commands/runtime.py` as the ratcheted hotspots for runtime orchestration, container wiring, and user-facing local serve or start flow.
- Treat `docker_cli.py` as a thin compatibility seam; do not move new runtime flow back into it.
- Treat `models.py`, `config_migration.py`, and `validator.py` as config-contract hotspots whose ownership should stay split between schema, compatibility, and semantic validation.

## Change Rules

- Move docker image resolution, container wiring, readiness helpers, and platform-specific support into adjacent modules instead of growing `core.py` or `commands/runtime.py`.
- Prefer extraction-first edits when adding new serve/start/status behavior.
- Keep `docker_cli.py` as a re-export or compatibility layer; if runtime ownership needs to grow, grow `core.py`, `commands/runtime.py`, or a new adjacent support module instead.
- Keep parser/schema/merger logic out of runtime orchestration files unless the change truly belongs to config translation.
- Do not encode the same canonical field inventory or signal/plugin type list independently across `models.py`, `config_migration.py`, and `validator.py` when one shared inventory or helper can own it.
- When adding new config-contract surface area, prefer dedicated schema-family or validation helpers instead of growing the all-in-one `models.py` / `validator.py` hotspots.
- Keep legacy migration behavior behind explicit compatibility helpers; do not turn `config_migration.py` into a second general-purpose config runtime.
