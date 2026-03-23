# OpenClaw-to-VSR Install and Import Workstream Execution Plan

## Goal

- Add one durable, repo-native workstream for installing vLLM Semantic Router from an agent prompt, shipping an OpenClaw bridge skill, and importing existing OpenClaw model providers into canonical VSR config.
- Keep the implementation English only across site copy, docs, skill metadata, CLI help, plan text, commit messages, and PR text.
- Close the workstream only after the website/docs surface, skill-pack surface, CLI importer, validations, and final PR state all agree on the same contract.

## Scope

- `docs/agent/plans/**`
- `website/src/components/InstallQuickStartSection/**`
- `website/docs/installation/**`
- `dashboard/backend/config/openclaw-skills.json`
- bundled OpenClaw skill-pack assets under `skills/**`
- image packaging paths that expose `/app/skills/**`
- `src/vllm-sr/cli/**`
- focused Python and asset-regression tests under `src/vllm-sr/tests/**`
- nearest local rules for `src/vllm-sr/cli/**`

## Exit Criteria

- The homepage install section exposes English `Human` and `Agent` modes while keeping the supported human one-liner unchanged.
- Installation docs document the agent-safe installer prompts, explain when to use them, and point to the repo-managed `openclaw-vsr-bridge` skill as the preferred long-term path.
- The repository ships one publishable OpenClaw-compatible skill bundle named `openclaw-vsr-bridge`, the OpenClaw skill catalog lists it, and packaged images expose the matching `SKILL.md`.
- `vllm-sr config import --from openclaw` imports supported OpenClaw provider/model endpoints into canonical `config.yaml`, preserves unrelated target config sections, bootstraps a minimal canonical target when needed, and rewrites the imported OpenClaw config to the first VSR listener only after the target write succeeds.
- Focused tests cover import success, duplicate model IDs across providers, merge/bootstrap behavior, rewrite safety, malformed or unsupported OpenClaw inputs, and the skill catalog asset contract.
- Final repo-native validations for the full changed-file set pass and one English PR is opened from the feature branch.

## Task List

- [x] `OCV001` Create the indexed execution plan, register it in the plan index, and lock the branch and harness entrypoints for the workstream.
- [x] `OCV002` Implement L1 website/docs agent install surfaces without changing installer semantics.
- [x] `OCV003` Implement L2 repo-managed OpenClaw skill-pack assets, catalog metadata, and packaging references.
- [x] `OCV004` Implement L3 `vllm-sr config import --from openclaw` with focused tests and CLI/docs discoverability.
- [x] `OCV005` Run the final validation ladder for the full changed-file set, update this plan with outcomes, and open one English PR.

## Current Loop

- Date: 2026-03-23
- Current task: `OCV005` completed with one external E2E blocker recorded in the validation log
- Branch: `vsr/openclaw-vsr-install-import`
- Planned loop order:
  - `L1` homepage install surface plus installation docs
  - `L2` repo-managed OpenClaw bridge skill plus catalog/package exposure
  - `L3` CLI import command plus focused tests and docs/help sync
  - final validation, signed-off commits, and one English PR
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/{README.md,repo-map.md,environments.md,change-surfaces.md,plans/README.md}`
  - nearest local rules read for `src/vllm-sr/cli/AGENTS.md`
  - broad `codebase-retrieval` across website install UI/docs, OpenClaw skill catalog and packaging, CLI config commands/helpers, dashboard OpenClaw config generation, tests, and Make targets
  - `make agent-report ENV=cpu CHANGED_FILES="dashboard/backend/Dockerfile,dashboard/backend/config/openclaw-skills.json,docs/agent/plans/README.md,docs/agent/plans/pl-0013-openclaw-vsr-install-import-workstream.md,skills/openclaw-vsr-bridge/SKILL.md,src/vllm-sr/Dockerfile,src/vllm-sr/Dockerfile.rocm,src/vllm-sr/README.md,src/vllm-sr/cli/commands/config.py,src/vllm-sr/cli/commands/general.py,src/vllm-sr/cli/config_import.py,src/vllm-sr/tests/test_config_import_openclaw.py,src/vllm-sr/tests/test_openclaw_skill_catalog.py,tools/agent/repo-manifest.yaml,website/docs/installation/configuration.md,website/docs/installation/installation.md,website/src/components/InstallQuickStartSection/index.module.css,website/src/components/InstallQuickStartSection/index.tsx"`
  - shell discovery with `sed`, `rg`, `find`, and `git status` across the plan index, install docs/UI, CLI command layout, dashboard OpenClaw handlers, Docker packaging, and test surfaces
  - `python -m pytest src/vllm-sr/tests/test_config_import_openclaw.py src/vllm-sr/tests/test_openclaw_skill_catalog.py -q` passed
  - `python3 -m ruff check --config tools/linter/python/.ruff.toml src/vllm-sr/cli/commands/config.py src/vllm-sr/cli/commands/general.py src/vllm-sr/cli/config_import.py src/vllm-sr/tests/test_config_import_openclaw.py src/vllm-sr/tests/test_openclaw_skill_catalog.py` passed
  - `make docs-lint` passed
  - `make docs-build` passed
  - `make dashboard-check` passed
  - `make agent-validate` passed
  - `make vllm-sr-test` passed
  - `make agent-ci-gate CHANGED_FILES="dashboard/backend/Dockerfile,dashboard/backend/config/openclaw-skills.json,docs/agent/plans/README.md,docs/agent/plans/pl-0013-openclaw-vsr-install-import-workstream.md,skills/openclaw-vsr-bridge/SKILL.md,src/vllm-sr/Dockerfile,src/vllm-sr/Dockerfile.rocm,src/vllm-sr/README.md,src/vllm-sr/cli/commands/config.py,src/vllm-sr/cli/commands/general.py,src/vllm-sr/cli/config_import.py,src/vllm-sr/tests/test_config_import_openclaw.py,src/vllm-sr/tests/test_openclaw_skill_catalog.py,tools/agent/repo-manifest.yaml,website/docs/installation/configuration.md,website/docs/installation/installation.md,website/src/components/InstallQuickStartSection/index.module.css,website/src/components/InstallQuickStartSection/index.tsx"` passed
  - `make agent-feature-gate ENV=cpu CHANGED_FILES="dashboard/backend/Dockerfile,dashboard/backend/config/openclaw-skills.json,docs/agent/plans/README.md,docs/agent/plans/pl-0013-openclaw-vsr-install-import-workstream.md,skills/openclaw-vsr-bridge/SKILL.md,src/vllm-sr/Dockerfile,src/vllm-sr/Dockerfile.rocm,src/vllm-sr/README.md,src/vllm-sr/cli/commands/config.py,src/vllm-sr/cli/commands/general.py,src/vllm-sr/cli/config_import.py,src/vllm-sr/tests/test_config_import_openclaw.py,src/vllm-sr/tests/test_openclaw_skill_catalog.py,tools/agent/repo-manifest.yaml,website/docs/installation/configuration.md,website/docs/installation/installation.md,website/src/components/InstallQuickStartSection/index.module.css,website/src/components/InstallQuickStartSection/index.tsx"` initially failed because the local Docker daemon was down
  - reran `make agent-feature-gate ...` after starting Docker Desktop and then with `PIP_TRUSTED_HOST='pypi.org files.pythonhosted.org'`; `make vllm-sr-test-integration` passed and `make agent-smoke-local` passed
  - `make e2e-test E2E_PROFILE=ai-gateway E2E_VERBOSE=true` failed once on an external chart download timeout for Bitnami `redis`, then retried and advanced past chart fetch but stalled in profile bootstrap while the `semantic-router` pod remained unready during unauthenticated first-run Hugging Face model downloads; pod logs showed no OpenClaw-to-VSR code-path failure before the retry was stopped
  - follow-up website/docs loop for hosted agent prompt files: `make agent-report ENV=cpu CHANGED_FILES="website/src/components/InstallQuickStartSection/index.tsx,website/docs/installation/installation.md,website/static/install/agent/vllm-sr-cli.md,website/static/install/agent/openclaw-vsr-bridge.md"`, `make docs-lint`, and `make docs-build` all passed

## Decision Log

- 2026-03-23: This workstream uses one final PR, not stacked PRs.
- 2026-03-23: The repo already exposes an OpenClaw skill catalog plus `/app/skills/<id>/SKILL.md` runtime lookup, so the bridge skill should plug into that existing packaging contract instead of inventing a second skill loader.
- 2026-03-23: The CLI importer should live in the existing `vllm-sr config ...` family, with extraction into dedicated helper modules instead of growing `config_migration.py` into a second general-purpose runtime.
- 2026-03-23: Agent install copy should keep the supported installer URL but force the agent-safe path with `--mode cli --runtime skip --no-launch` so automation does not auto-start serve or open a browser during handoff.
- 2026-03-23: The repo-managed bridge skill is shipped through the existing `dashboard/backend/config/openclaw-skills.json` plus `/app/skills/<id>/SKILL.md` packaging contract, so the runtime images now copy the top-level `skills/` tree into `/app/skills/`.
- 2026-03-23: When duplicate OpenClaw model IDs appear across providers, the importer keeps the upstream `provider_model_id`, assigns a stable logical VSR name as `<provider>/<model_id>`, and rewrites OpenClaw references only after the VSR target write succeeds.
- 2026-03-23: Local editable-install TLS failures in the feature gate were resolved with the environment-only retry `PIP_TRUSTED_HOST='pypi.org files.pythonhosted.org'`; no repository code change was needed for that workstation-specific network issue.
- 2026-03-23: The homepage agent cards now copy short fetch-style prompts only; the full agent workflows are hosted as Markdown under `website/static/install/agent/*.md` so agents can fetch instructions from the website domain directly.

## Follow-up Debt / ADR Links

- No durable architecture debt was added.
- External validation blocker only: the local `ai-gateway` E2E retry advanced past Helm chart dependency fetch but then stalled on first-run model bootstrap in the Kind cluster while unauthenticated Hugging Face downloads kept the `semantic-router` startup probe closed on `:50051`.
