# Agent Report Local Artifact Loop

## Goal

- Add a repo-native local artifact path for `agent-report` outputs without turning those machine-generated files into canonical harness state.
- Keep the shared harness clear about which artifacts belong in gitignored local storage versus committed governance docs.

## Scope

- `tools/agent/scripts/agent_gate.py`
- `tools/agent/scripts/agent_models.py`
- new helper logic under `tools/agent/scripts/`
- `tools/make/agent.mk`
- `.gitignore`
- `.dockerignore`
- `docs/agent/governance.md`
- `docs/agent/context-management.md`
- `docs/agent/plans/README.md`

## Exit Criteria

- `agent-report` can write a local artifact on demand without changing its existing stdout behavior.
- The default local artifact layout lives under a dedicated gitignored directory instead of `.vllm-sr/`.
- Harness docs clearly distinguish local generated artifacts from committed execution plans, ADRs, and debt entries.
- Canonical plan indexing and manifest governance stay aligned with the new loop.

## Task List

- [x] `AR001` Create and index a canonical execution plan for this workstream.
  - Done when this file is added under `docs/agent/plans/` and indexed from the plan README and repo manifest.
- [x] `AR002` Add explicit write support to `agent-report`.
  - Done when `tools/agent/scripts/agent_gate.py report` can write the JSON payload to a requested local path without breaking existing stdout output.
- [x] `AR003` Define the default local artifact layout under `.agent-harness/`.
  - Done when the harness has a documented default path for the latest report plus session-stamped report captures and the directory is gitignored.
- [x] `AR004` Clarify the governance boundary between local artifacts and canonical harness state.
  - Done when governance/context docs say that `.agent-harness/` is gitignored local state while plans, ADRs, and debt entries remain the committed source of truth.
- [ ] `AR005` Record a clean repo-native validation rerun for the landed artifact contract.
  - Done when the applicable repo-native validation commands pass on a clean rerun and the results are recorded here without treating transient workstation or dependency-fetch failures as durable harness state.

## Current Loop

- 2026-03-26: `AR001` created this execution plan and indexed it in the canonical plan inventory so the work can resume from the repo alone.
- 2026-03-26: `AR002` added `agent-report` write support through explicit local artifact flags instead of changing the default stdout contract.
- 2026-03-26: `AR003` defined `.agent-harness/reports/latest-report.json` plus timestamped session captures under `.agent-harness/reports/sessions/`, and the new top-level directory is now ignored by git and Docker build context.
- 2026-03-26: `AR004` updated governance and context docs so local machine-generated artifacts are clearly separated from committed execution plans, ADRs, and debt entries.
- 2026-03-26: `AR005` reached a clean lightweight validation baseline with `make agent-validate` and `make agent-ci-gate CHANGED_FILES="..."` passing for the touched harness files.
- 2026-03-26: `AR005` also verified both new write entrypoints with `make agent-report AGENT_REPORT_WRITE=1 ...` and `make agent-report AGENT_REPORT_WRITE_PATH=.agent-harness/reports/custom.json ...`.
- 2026-03-26: `AR005` is still open for full completion-gate closure. `make agent-feature-gate ENV=cpu CHANGED_FILES="..."` entered the expected `make agent-dev` local-build path, but the user explicitly deferred the heavy local build, serve, smoke, and affected E2E chain before it completed.
- 2026-03-26: A later PR-baseline attempt reached `make agent-pr-gate`, but that run hit a transient `proxy.golang.org` dependency-fetch failure during `golang-lint`; treat that as historical environment noise, not as durable evidence that the landed artifact contract is still wrong.

## Decision Log

- Keep local report artifacts outside `.vllm-sr/` so harness-generated session state does not blur into CLI or dashboard workspace state.
- Preserve the existing `agent-report` stdout contract; artifact writing is opt-in and additive.
- Treat `.agent-harness/` as restart-safe local state only. Canonical resumable loop state still belongs in `docs/agent/plans/*.md`.
- Keep the first artifact format narrow: JSON report payloads plus stable file layout, without introducing a second planning or debt source of truth.

## Follow-up Debt / ADR Links

- [README.md](README.md)
- [../tech-debt-register.md](../tech-debt-register.md)
- [../tech-debt/README.md](../tech-debt/README.md)
- [../adr/README.md](../adr/README.md)
