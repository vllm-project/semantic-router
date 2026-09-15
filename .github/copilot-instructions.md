# GitHub Copilot instructions

Read `AGENTS.md` and the nearest local `AGENTS.md` for changed directories.
Use `make impact CHANGED_FILES="..."` for ownership and candidate checks, then
`make check CHANGED_FILES="..."` for the daily deterministic gate. Select
integration or E2E explicitly with `make verify DOMAIN=...` or
`make verify PROFILE=...`; use `make harness-check` for harness/workflow edits.

Prioritize correctness, security, public contracts, missing behavior tests, and
real dependency boundaries. Numeric size metrics are advisory: recommend an
extraction only when it improves ownership, coupling, or testability.

Behavior-visible routing, startup, config, Docker, CLI, API, or protocol
changes need appropriate integration evidence; pure refactors do not. Keep
suggestions inside the requested subsystem and never copy credentials, private
infrastructure, local receipts, or AI/tool attribution into repository files.
