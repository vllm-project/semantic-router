# GitHub Copilot Instructions

Use the repository harness for reviews and code suggestions.

## Start Here

1. Read [`AGENTS.md`](../AGENTS.md) and
   [`tools/agent/docs/README.md`](../tools/agent/docs/README.md).
2. Resolve the task before proposing changes:

   ```bash
   make agent-report ENV=cpu CHANGED_FILES="path/to/file ..."
   ```

3. Follow the reported primary skill, context, nearest local `AGENTS.md`, and
   validation commands.

`tools/agent/docs/` is the human-readable contract. Manifests, scripts, Make
targets, and workflows are the executable contract. Keep them aligned instead
of restating their rules here.

## Review Priorities

Report concrete, file-specific findings in this order:

1. correctness, security, and behavior regressions;
2. public API or configuration compatibility;
3. missing tests or affected E2E coverage;
4. module-boundary and hotspot growth;
5. drift between documentation and executable rules.

Behavior-visible routing, startup, config, Docker, CLI, or API changes normally
need E2E coverage. Harness changes need `make agent-validate`. Use the full gate
reported by `make agent-report` before describing a change as complete.

Keep suggestions within the requested subsystem, preserve DCO sign-off, and do
not copy branch notes, AI/tool attribution, credentials, private paths, or test
receipts into durable documentation.
