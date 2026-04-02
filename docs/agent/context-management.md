# Context Management

This document defines how the harness exposes the minimum useful context for a task instead of forcing agents to read the entire `docs/agent/` tree.

## Why This Exists

- `AGENTS.md` should stay short and navigational.
- `docs/agent/*` should stay canonical, but the set is now large enough that agents need a task-first read path.
- `make agent-report ENV=cpu|amd CHANGED_FILES="..."` should resolve not only validation, but also the minimum context pack and loop policy for the task.

## Disclosure Layers

- `L0` entrypoint
  - `AGENTS.md`
  - `docs/agent/README.md`
- `L1` task contract
  - resolved primary skill
  - fragment skills
  - loop mode and execution-plan guidance for the active task
  - the `## Must Read` links referenced by those skills
- `L2` surface context
  - only docs and executable sources for the impacted surfaces
- `L3` hotspot supplements
  - nearest local `AGENTS.md` files for changed hotspot trees
- `L4` durable loop context
  - execution plans, ADRs, and tech debt only when the task needs resumable or unresolved context

## Context Pack Flow

1. Resolve changed files through `make agent-report ENV=cpu|amd CHANGED_FILES="..."` so the harness can emit the active skill, loop mode, execution-plan guidance, and validation commands.
2. Select the primary skill and fragment skills from `tools/agent/skill-registry.yaml`.
3. Pull the skill `## Must Read` references from the active `SKILL.md` files.
4. Add surface-specific references from `tools/agent/context-map.yaml`.
5. Add nearest local `AGENTS.md` files for hotspot paths when applicable.
6. Add resume references such as plans or debt as low-priority follow-up context, then promote them into the active loop when the task becomes long-horizon, multi-loop, or unresolved.

Repo-native convenience:

- `make agent-report AGENT_REPORT_WRITE=1 ENV=cpu|amd CHANGED_FILES="..."`
  - writes `.agent-harness/reports/latest-report.json` plus a timestamped session copy under `.agent-harness/reports/sessions/`
- `make agent-report AGENT_REPORT_WRITE_PATH=.agent-harness/reports/custom.json ...`
  - writes one explicit local JSON artifact
- `make agent-report AGENT_REPORT_CONTEXT_DETAIL=full ...`
  - expands the printed summary to show the full context-pack sections instead of the compact default

These files are gitignored local artifacts derived from the canonical harness. They are useful for local handoff or resume hints, but they are not a second source of truth.

## Source of Truth

- Human-readable policy: this document and the linked harness docs under `docs/agent/`
- Executable map: `tools/agent/context-map.yaml`
- Skill-owned required reading: the active `SKILL.md` files under `tools/agent/skills/`
- Runtime assembly: `tools/agent/scripts/agent_context_pack.py`
- Validation: `make agent-validate`

## Default Budget

- The default `agent-report` summary stays compact.
- The printed summary should prioritize:
  - `Start here`
  - the primary skill plus active fragments only
  - the smallest useful `Must Read` set
- Full `read_if_applicable` and resume references remain available through `--context-detail full` or `--format json`.
- Execution-plan resume references belong in the default pack only when the resolved task still needs long-horizon loop state.
- Debt registers, catalogs, and other index-style docs should not be injected into ordinary task context unless the current task actually needs them.

## Maintenance Rules

- Do not duplicate full guidance into the context map; point to the canonical doc or executable file.
- Keep the context pack task-first and minimal; if a reference is almost always skipped, remove it or lower it to resume-only context.
- When a new surface, skill, or local rule is added, update the context map in the same change.
- If the context pack and the canonical docs disagree, fix the canonical doc and the executable map together.
- Keep `.agent-harness/` outputs gitignored and derived; if loop state needs to be shared or versioned, promote it into an execution plan, ADR, or debt entry instead.
