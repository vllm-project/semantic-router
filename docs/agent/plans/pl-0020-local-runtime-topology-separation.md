# Local Runtime Topology Separation Workstream

## Goal

- Split the current local `vllm-sr-container` runtime topology so dashboard, router, and Envoy run as three managed containers on the shared `vllm-sr-network`.
- Preserve the canonical `vllm-sr serve` experience, existing host-facing ports, and stack-name/port-offset behavior while removing the single-container startup assumption.
- Keep connectivity intact across dashboard, router, Envoy, OpenClaw agents, fleet-sim, and observability sidecars, and preserve the config update path from dashboard writes through router reload and Envoy regeneration.

## Scope

- `src/vllm-sr/**`
- `src/vllm-sr/Dockerfile*`
- `dashboard/backend/**`
- focused router runtime reload references under `src/semantic-router/pkg/extproc/**`
- focused runtime/OpenClaw tests under `src/vllm-sr/tests/**`
- `docs/agent/plans/**`
- executable harness indexing for the new plan in `tools/agent/repo-manifest.yaml`
- nearest local rules for `src/vllm-sr/cli/**` and `dashboard/backend/handlers/**`

## Exit Criteria

- `vllm-sr serve --image-pull-policy never` starts dashboard, router, and Envoy as separate managed containers attached to the same stack-scoped bridge network.
- The split topology preserves the current host-facing contract for dashboard, router API, metrics, and configured listener ports, including stack-name and port-offset isolation.
- Dashboard backend reaches router API, router metrics, and Envoy through explicit cross-container service URLs instead of implicit same-container `localhost` assumptions.
- Dashboard config update, deploy, rollback, and setup-activation flows still write canonical config, sync the runtime override config, preserve router hot reload, and regenerate or restart Envoy successfully.
- OpenClaw bridge-network provisioning, gateway reachability checks, embedded dashboard proxying, and model-base-url rewriting continue to work when dashboard, router, and Envoy are no longer co-located in one container.
- Targeted tests, CPU-local smoke, CLI fast or integration validation, and any affected local E2E profile selected by `make agent-e2e-affected` pass for the final changed-file set.

## Task List

- [x] `TS001` Create the indexed execution plan, register it in the harness indexes, and capture the current topology invariants plus the non-negotiable connectivity and config-update constraints.
- [x] `TS002` Define the split runtime naming, shared-network, volume-mount, and host-port contract for dashboard, router, and Envoy under `RuntimeStackLayout`.
- [x] `TS003` Refactor CLI startup, shutdown, logs, and status orchestration to manage the split containers without introducing a second local serve path.
- [x] `TS004` Refactor dashboard runtime assumptions and runtime-apply helpers so cross-container router and Envoy control works without relying on single-container `supervisorctl` shortcuts.
- [x] `TS005` Refactor OpenClaw bridge-mode defaults and service discovery so OpenClaw workers continue to reach the routed model endpoint and embedded dashboard proxy paths after the split.
- [x] `TS006` Add or update focused tests plus smoke or E2E coverage for split-topology startup, connectivity, and config propagation.
- [x] `TS007` Run the validation ladder for the final changed-file set, update this plan with results, and record any remaining durable debt if the architecture still diverges.

## Current Loop

- Date: 2026-03-25
- Current task: post-`TS007` compatibility cleanup complete for memory-workflow CI handling, Docker-only CLI validation harness alignment, and dashboard OpenClaw runtime detection; no new runtime or connectivity gap is open in the active local sweep
- Branch: `codex/topology-separation-research`
- Planned loop order:
  - `L1` capture current single-container topology, cross-container dependencies, and validation boundaries
  - `L2` split runtime naming and startup orchestration under the canonical CLI path
  - `L3` split dashboard runtime-control and OpenClaw assumptions while preserving config propagation
  - `L4` expand tests, run smoke or E2E coverage, and close or record residual debt
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/{README.md,repo-map.md,environments.md,change-surfaces.md,feature-complete-checklist.md,plans/README.md}`, `.agents/skills/harness/SKILL.md`, `src/vllm-sr/cli/AGENTS.md`, and `dashboard/backend/handlers/AGENTS.md`
  - broad repository retrieval with `rg` and focused file reads across `src/vllm-sr/**`, `dashboard/backend/**`, `src/semantic-router/pkg/extproc/server.go`, Dockerfiles, tests, and OpenClaw runtime files because the dedicated `codebase-retrieval` tool was not available in this session
  - `make agent-report ENV=cpu CHANGED_FILES="src/vllm-sr/cli/docker_cli.py src/vllm-sr/supervisord.conf src/vllm-sr/start-dashboard.sh dashboard/backend/handlers/deploy.go dashboard/backend/handlers/config.go dashboard/backend/handlers/status.go docs/agent/plans/pl-0020-local-runtime-topology-separation.md"`
  - `go test ./handlers -run 'Test(ManagedContainerNameForService|ManagedRuntimeUsesSplitContainers|ManagedContainerNamesForComponentAll|ManagedServiceForContainerName|ConfiguredRuntimeConfigPathUsesEnvOverride|SyncRuntimeConfigLocallyWritesInternalRuntimeConfig|RuntimeSyncPythonBinaryRejectsNonPythonOverride|DefaultOpenClawModelBaseURL|ResolveOpenClawModelBaseURL|DetectRouterRuntimeStatus)'` from `dashboard/backend`
  - `python3 -m pytest tests/test_openclaw_shared_network.py -q` from `src/vllm-sr`
  - `python3 -m pytest tests/test_deployment_backend.py -q` from `src/vllm-sr`
  - `bash -n src/vllm-sr/start-router.sh && sh -n src/vllm-sr/start-envoy.sh && sh -n src/vllm-sr/start-dashboard.sh`
  - `go test ./handlers -run 'Test(CollectHostStatusPrefersSplitManagedRuntimeOverLegacyContainerResidue|ManagedContainerNameForService|ManagedRuntimeUsesSplitContainers|ManagedContainerNamesForComponentAll|ManagedServiceForContainerName|ConfiguredRuntimeConfigPathUsesEnvOverride|SyncRuntimeConfigLocallyWritesInternalRuntimeConfig|RuntimeSyncPythonBinaryRejectsNonPythonOverride|DefaultOpenClawModelBaseURL|ResolveOpenClawModelBaseURL|DetectRouterRuntimeStatus)'` from `dashboard/backend`
  - `make dashboard-check`
  - `go test ./handlers -run 'Test(RuntimeContainerStatusForLogsPrefersSplitManagedRuntimeOverLegacyContainerResidue|CollectHostStatusPrefersSplitManagedRuntimeOverLegacyContainerResidue|OpenClawModelGatewayContainerNamePrefersExplicitOverride|OpenClawModelGatewayContainerNameDerivesFromTargetEnvoyURL|OpenClawModelGatewayContainerNameFallsBackToManagedEnvoyContainer|ManagedContainerNameForService|ManagedRuntimeUsesSplitContainers|ManagedContainerNamesForComponentAll|ManagedServiceForContainerName|ConfiguredRuntimeConfigPathUsesEnvOverride|SyncRuntimeConfigLocallyWritesInternalRuntimeConfig|RuntimeSyncPythonBinaryRejectsNonPythonOverride|DefaultOpenClawModelBaseURL|ResolveOpenClawModelBaseURL|DetectRouterRuntimeStatus)'` from `dashboard/backend`
  - `make dashboard-check`
  - `make agent-ci-gate CHANGED_FILES="dashboard/backend/handlers/logs.go dashboard/backend/handlers/logs_test.go dashboard/backend/handlers/openclaw_helpers.go dashboard/backend/handlers/openclaw_provision.go dashboard/backend/handlers/openclaw_test.go dashboard/backend/handlers/status_collectors.go dashboard/backend/handlers/status_collectors_test.go docs/agent/plans/pl-0020-local-runtime-topology-separation.md"` (blocked by unrelated existing `src/semantic-router` binding type-check failures in `ml-binding` and `candle-binding`)
  - manual local image rebuild for `ghcr.io/vllm-project/semantic-router/vllm-sr:latest` after updating split-runtime entrypoints and Envoy DNS-aware config generation
  - `python3 -m pytest tests/test_openclaw_shared_network.py tests/test_docker_runtime.py tests/test_deployment_backend.py tests/test_config_generator.py -q` from `src/vllm-sr`
  - `go test ./pkg/modeldownload -run 'TestBuildModelSpecs(SkipsRouterOwnedDefaultsForAgentSmokeConfigs|IncludesRouterOwnedDefaultsForScratchCanonicalConfig|IncludesAllAMDDeployModels|SkipsDisabledHallucinationFeatureModels|SkipsUnusedFeedbackDetectorDefaults)' -count=1` from `src/semantic-router`
  - `go test ./pkg/config -run TestMaintainedConfigAssetsUseCanonicalV03Contract -count=1` from `src/semantic-router`
  - `make agent-serve-local ENV=cpu AGENT_STACK_NAME=topology-smoke AGENT_PORT_OFFSET=200`
  - `make agent-smoke-local AGENT_STACK_NAME=topology-smoke AGENT_PORT_OFFSET=200`
  - `make agent-e2e-affected CHANGED_FILES="dashboard/backend/handlers/logs.go dashboard/backend/handlers/openclaw.go dashboard/backend/handlers/openclaw_helpers.go dashboard/backend/handlers/openclaw_provision.go dashboard/backend/handlers/openclaw_test.go dashboard/backend/handlers/runtime_config_apply.go dashboard/backend/handlers/runtime_config_sync.go dashboard/backend/handlers/runtime_config_sync_test.go dashboard/backend/handlers/setup.go dashboard/backend/handlers/status_collectors.go dashboard/backend/handlers/status_runtime.go e2e/config/config.agent-smoke.amd.yaml e2e/config/config.agent-smoke.cpu.yaml src/semantic-router/pkg/modeldownload/config_parser_test.go src/vllm-sr/Dockerfile src/vllm-sr/Dockerfile.rocm src/vllm-sr/cli/commands/runtime.py src/vllm-sr/cli/config_generator.py src/vllm-sr/cli/core.py src/vllm-sr/cli/docker_backend.py src/vllm-sr/cli/docker_runtime.py src/vllm-sr/cli/docker_start.py src/vllm-sr/cli/runtime_lifecycle.py src/vllm-sr/cli/runtime_stack.py src/vllm-sr/cli/templates/envoy.template.yaml src/vllm-sr/start-dashboard.sh src/vllm-sr/start-router.sh src/vllm-sr/start-envoy.sh src/vllm-sr/tests/test_openclaw_shared_network.py tools/make/agent.mk tools/agent/repo-manifest.yaml dashboard/backend/handlers/logs_test.go dashboard/backend/handlers/runtime_managed_services.go dashboard/backend/handlers/runtime_managed_services_test.go dashboard/backend/handlers/status_collectors_test.go src/vllm-sr/tests/test_config_generator.py src/vllm-sr/tests/test_docker_runtime.py"` (returned `No affected local E2E profiles.`)
  - `make agent-ci-gate CHANGED_FILES="src/vllm-sr/cli/core.py src/vllm-sr/tests/test_openclaw_shared_network.py tools/make/agent.mk e2e/config/config.agent-smoke.cpu.yaml e2e/config/config.agent-smoke.amd.yaml src/semantic-router/pkg/modeldownload/config_parser_test.go"` (still blocked by the same pre-existing `ml-binding` / `candle-binding` type-check failures; `black` reformatted the touched Python files and the focused pytest rerun stayed green)
  - `./.venv-codex/bin/python -m pip install --upgrade pip`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" python3 -m pip install -e src/vllm-sr`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make vllm-sr-test`
  - `cd e2e/testing/vllm-sr-cli && PATH="$(git rev-parse --show-toplevel)/.venv-codex/bin:$PATH" CONTAINER_RUNTIME=docker VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_SIM_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-sim:latest RUN_INTEGRATION_TESTS=true python run_cli_tests.py --verbose --integration`
  - `make agent-report ENV=cpu CHANGED_FILES=".github/workflows/integration-test-memory.yml"`
  - `python3 - <<'PY' ... yaml.safe_load(Path('.github/workflows/integration-test-memory.yml').read_text()) ... PY`
  - `make agent-validate`
  - `make agent-e2e-affected CHANGED_FILES=".github/workflows/integration-test-memory.yml"` (blocked locally because `kind` is not installed, so the required `ai-gateway` profile could not start its test cluster)
  - `python3 -m pytest e2e/testing/vllm-sr-cli/test_unit_runtime_topology.py -q`
  - `python3 -m pytest e2e/testing/vllm-sr-cli/test_unit_serve.py -q`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make vllm-sr-test`
  - `cd e2e/testing/vllm-sr-cli && PATH="$(git rev-parse --show-toplevel)/.venv-codex/bin:$PATH" CONTAINER_RUNTIME=docker VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_SIM_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-sim:latest RUN_INTEGRATION_TESTS=true python run_cli_tests.py --verbose --integration`
  - `make agent-serve-local ENV=cpu AGENT_STACK_NAME=docker-only-smoke AGENT_PORT_OFFSET=300`
  - `make agent-smoke-local AGENT_STACK_NAME=docker-only-smoke AGENT_PORT_OFFSET=300`
  - `VLLM_SR_STACK_NAME=docker-only-smoke VLLM_SR_PORT_OFFSET=300 vllm-sr stop`
  - `go test ./handlers -run 'Test(EnsureImageAvailable_AlwaysPullsRemoteImage|ResolveBaseImage_KeepsDefaultInsteadOfAutoSelectingLocalFallback|EnsureImageAvailable_LocalTagStillRequiresLocalImage|DetectContainerRuntimeUsesExplicitRuntimePath|DetectContainerRuntimeRejectsPodmanOverride|DetectContainerRuntimeRejectsPodmanContainerRuntimeEnv)'` from `dashboard/backend`
  - `make dashboard-check`
- Key findings:
  - The local runtime still centers on one `vllm-sr-container` image entrypoint that starts router, Envoy, and dashboard through `supervisord`.
  - The stack-scoped bridge network already exists as a first-class runtime contract for OpenClaw, fleet-sim, and observability; the split should extend that network rather than invent a second topology model.
  - Dashboard runtime defaults still assume co-location: `start-dashboard.sh` points router API, router metrics, and Envoy at `localhost`, status collectors key on the single `vllm-sr-container`, and several setup or runtime-apply helpers call `docker exec vllm-sr-container ...`.
  - Router and Envoy do not reload the same way today: router watches the synced runtime config file and hot-reloads via fsnotify, while dashboard explicitly regenerates Envoy config and restarts the Envoy process after config writes.
  - OpenClaw bridge-mode provisioning already depends on the shared network, but its loopback rewrite path still falls back to the dashboard or `vllm-sr-container` name as the default upstream target for routed model traffic.
  - `RuntimeStackLayout` now defines role-specific router, Envoy, and dashboard container names plus service URLs without changing the legacy `vllm-sr-container` contract yet.
  - Dashboard backend runtime sync, runtime apply, setup restart, status collection, and log retrieval now resolve managed container names dynamically so split-container control can be introduced without breaking the current single-container defaults.
  - OpenClaw model gateway defaults now prefer `TARGET_ENVOY_URL` and can rewrite bridge-mode loopback traffic toward an explicit model-gateway container instead of assuming dashboard and the routed model endpoint are co-located.
  - CLI `stop`, `status`, and `logs` now probe role-specific runtime containers first and fall back to the legacy `vllm-sr-container`, so operational entrypoints can survive the future startup split.
  - The local startup chain now has an initial split implementation path: the CLI prepares shared mounts and service URLs, launches router, Envoy, and dashboard as separate containers from the same image via role-specific entrypoints, and routes Envoy extproc or API back to router through service-name injection instead of hardcoded same-container loopback.
  - Dashboard backend host-mode status collection now prioritizes split managed runtime containers over stale legacy `vllm-sr-container` residue, so status reporting does not regress to the old topology when both naming schemes are present on the same Docker host.
  - Dashboard backend log selection now follows the same split-runtime precedence as status collection, so stale `vllm-sr-container` residue does not mask router, Envoy, or dashboard logs from the active split deployment.
  - OpenClaw bridge-mode loopback rewriting now resolves the model gateway container through explicit override, `TARGET_ENVOY_URL`, or the managed Envoy container name before considering legacy fallbacks, which keeps routed model traffic aligned with the split topology.
  - The repository-level `agent-ci-gate` is currently blocked by unrelated existing `src/semantic-router` type-check issues in the binding packages rather than by this dashboard or OpenClaw slice; `make dashboard-check` and the focused handler tests remain green for the active change set.
  - Docker Desktop local startup no longer needs a host-mounted Docker CLI inside the dashboard container by default; the runtime image already carries `docker`, which avoids macOS file-sharing failures while preserving an explicit opt-in host CLI mount.
  - Envoy now starts cleanly in the split topology with router service-name upstreams because the generated extproc and API fallback clusters switch to DNS-aware cluster types when the upstream host is a container name instead of a literal IP.
  - The local smoke harness now recognizes the split router, Envoy, and dashboard containers instead of insisting on a single `vllm-sr-container`, so repo-native smoke validation matches the new topology.
  - `agent-serve-local` now passes an explicit workspace state root into the CLI, which keeps `.vllm-sr` state, OpenClaw registry data, and the `models/` cache anchored at the repo root instead of under `e2e/config/`.
  - The maintained smoke configs now explicitly clear router-owned default embedding or classifier modules and disable semantic cache, so CPU- and AMD-local smoke stays API-only and validates startup plus connectivity without spending minutes on unrelated default model downloads or BERT warmup.
  - CPU-local smoke now passes on an isolated split-runtime stack via `make agent-serve-local ENV=cpu AGENT_STACK_NAME=topology-smoke AGENT_PORT_OFFSET=200` and `make agent-smoke-local AGENT_STACK_NAME=topology-smoke AGENT_PORT_OFFSET=200`; router, Envoy, dashboard, and fleet-sim all report healthy together on the split topology.
  - The CLI integration harness under `e2e/testing/vllm-sr-cli` still assumed a single `vllm-sr-container`; split-runtime validation required teaching the shared test base to resolve legacy single-container and split router or Envoy or dashboard topologies, and to clean up observability sidecars too.
  - With that harness update, the full local CLI suite now passes on split runtime: `make vllm-sr-test` is green in a repo-local virtualenv, and the 32-test integration runner behind `make vllm-sr-test-integration` also passes end-to-end against split router, Envoy, dashboard, and fleet-sim containers.
  - The original local blocker on `make vllm-sr-test` or `make vllm-sr-test-integration` was not the topology work itself but the host `python3` shipping `pip 21.2.3`, which cannot install `src/vllm-sr` in editable `pyproject.toml` mode; a repo-local virtualenv with modern pip is enough to clear that validation path without changing the system Python install.
  - `make agent-e2e-affected` reports no affected local E2E profiles for this change set, so the profile map does not currently require an `ai-gateway` local E2E rerun on top of the passing smoke and CLI validation ladder.
  - The memory-integration GitHub workflow still assumed a legacy single `vllm-sr-container` for failure logs and cleanup after the split-runtime work landed; aligning those after-action steps is enough to keep the workflow useful without changing the memory integration runner itself.
  - The repo-native local E2E selector for the memory-workflow cleanup maps to `ai-gateway`, but this workstation cannot currently run that profile because `kind` is missing from `PATH`; that is an environment blocker rather than a regression from the CI workflow edit.
  - The CLI test harness under `e2e/testing/vllm-sr-cli` had one remaining contract mismatch after the runtime moved to Docker-only detection: its pre-flight and base-class runtime discovery still auto-accepted Podman. Aligning the runner and base class to reject Podman keeps validation semantics consistent with the real local serve path.
  - Re-running the lightweight CLI gates plus the full direct integration runner against the already-built split-runtime image is a better validation fit for this Docker-only harness cleanup than forcing a fresh `make vllm-sr-test-integration` image rebuild, because the current edits do not touch the runtime image contents.
  - CPU-local smoke still passes on an isolated stack after the CLI harness cleanup, so tightening the validation entrypoint to Docker-only did not regress split-runtime startup, health, or stop behavior.
  - Dashboard OpenClaw provisioning still had one stale Podman assumption after the broader Docker-only runtime hardening: its runtime auto-detection and user-facing error message still advertised `docker/podman`. Tightening that helper to reject Podman keeps dashboard-side provisioning expectations aligned with the local serve contract.

## Decision Log

- 2026-03-24: This workstream will extend the existing stack-scoped `vllm-sr-network` contract instead of introducing a compose-only or role-specific parallel network model.
- 2026-03-24: The first implementation target should preserve the canonical local image flow and reuse the existing runtime image, preferring role-specific container startup over immediately creating three independently versioned images.
- 2026-03-24: Host-facing dashboard, router API, metrics, and listener ports remain part of the public local-dev contract; topology separation should move behind `RuntimeStackLayout` instead of changing those endpoints up front.
- 2026-03-24: Router reload and Envoy reload must stay distinct in the split design: router keeps file-driven hot reload semantics, while Envoy keeps generated-config plus process-restart semantics until a narrower cross-container control seam exists.
- 2026-03-24: OpenClaw, dashboard proxies, and runtime status helpers should move toward explicit service-name configuration instead of relying on fallback assumptions that the dashboard and routed model endpoint live in the same container namespace.
- 2026-03-25: The repo-native smoke configs should stay API-only and avoid router-owned default model downloads; startup-chain validation for this workstream is about split-topology orchestration and connectivity, not cold-starting default local classifiers or semantic-cache BERT models.
- 2026-03-25: Three-image packaging is now tracked separately in [PL-0021](pl-0021-local-runtime-three-image-rollout.md) so the completed topology split can remain the compatibility baseline while packaging and publish contracts evolve independently.

## Follow-up Debt / ADR Links

- Related debt to watch while executing this plan:
  - [TD004](../tech-debt/td-004-python-cli-kubernetes-workflow-separation.md)
  - [TD031](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
  - [TD034](../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- Follow-on execution plan:
  - [PL-0021](pl-0021-local-runtime-three-image-rollout.md)
- No new durable debt entry was added in this loop; reopen or add one only if the split lands with unresolved architecture divergence that cannot be retired in the same workstream.
