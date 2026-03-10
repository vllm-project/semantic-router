# vllm-sr CLI Notes

## Scope

- `src/vllm-sr/cli/**`
- local rules for CLI orchestration and the `docker_cli.py` hotspot

## Responsibilities

- Keep CLI files centered on one dominant command-orchestration responsibility.
- Treat `docker_cli.py` as the ratcheted hotspot for top-level command flow and user-facing runtime decisions.

## Change Rules

- Move docker image resolution, container wiring, readiness helpers, and platform-specific support into adjacent modules instead of growing `docker_cli.py`.
- Prefer extraction-first edits when adding new serve/start/status behavior.
- Keep parser/schema/merger logic out of runtime orchestration files unless the change truly belongs to config translation.
