# Local Runtime Three-Image Rollout

## Status Note

- 2026-03-26 review follow-up contracted the rollout surface: split local runtime remains opt-in, router now reuses the existing `vllm-sr` / `vllm-sr-rocm` images, dashboard keeps the existing `dashboard` image, and Envoy now targets the upstream `envoyproxy/envoy` image with host-side config generation.
- Dedicated `vllm-sr-router*` and repo-managed `vllm-sr-envoy` publication are no longer part of the intended default contract for this plan, even though the historical loop notes below still capture the earlier exploration path.

## Goal

- Evolve the split local runtime from three role-specific containers that share one `vllm-sr` image into three role-specific images for router, Envoy, and dashboard.
- Preserve the canonical `vllm-sr serve` contract, stack-scoped `vllm-sr-network`, current host-facing ports, config propagation semantics, and OpenClaw connectivity while reducing image coupling and rebuild cost.
- Keep the rollout compatible with the current single-image transition state by introducing explicit per-role image contracts that fall back to the existing `VLLM_SR_IMAGE`.

## Scope

- `src/vllm-sr/Dockerfile*`
- new or refactored role-specific runtime Dockerfiles under `src/vllm-sr/**` as needed
- `dashboard/backend/Dockerfile`
- `src/vllm-sr/cli/**`
- `tools/make/docker.mk`
- `.github/workflows/docker-publish.yml`
- `.github/workflows/docker-release.yml`
- focused runtime and CLI validation under `src/vllm-sr/tests/**` and `e2e/testing/vllm-sr-cli/**`
- `docs/agent/plans/**`
- executable harness indexing for the new plan in `tools/agent/repo-manifest.yaml`

## Exit Criteria

- Local runtime image selection exposes explicit router, Envoy, and dashboard image contracts for `vllm-sr serve`, Make targets, and test harnesses while remaining backward compatible with `VLLM_SR_IMAGE`.
- `cpu-local` split runtime can start router, Envoy, and dashboard from independent images without changing host-facing dashboard, router API, router metrics, or Envoy listener URLs.
- `amd-local` keeps a first-class ROCm path for the router image while allowing dashboard and Envoy to remain on CPU-compatible images unless a narrower requirement appears.
- Dashboard and Envoy role images include exactly the runtime assets they need for config apply, status, OpenClaw, and local development flows instead of inheriting the full router runtime payload.
- Build and publish workflows can produce and tag the new role images without breaking the existing `vllm-sr`, `vllm-sr-rocm`, `dashboard`, or `vllm-sr-sim` contracts during the transition.
- Focused tests, `make agent-validate`, the relevant CLI test gates, and CPU-local smoke pass for the final changed-file set; any remaining architecture divergence is recorded as indexed debt.

## Task List

- [x] `TI001` Create the indexed execution plan, capture the current single-image evidence, and define the phased rollout contract.
- [x] `TI002` Introduce the per-role image contract in CLI, Make, and tests with full fallback to `VLLM_SR_IMAGE`.
- [x] `TI003` Carve out the Envoy runtime image and wire local startup, publish, and validation flows to use it.
- [x] `TI004` Carve out the dashboard runtime image with the Docker CLI, OpenClaw assets, and runtime helpers needed by local split-topology flows.
- [x] `TI005` Carve out the router runtime image, including the ROCm-specific path, and remove role payload that no longer belongs in the router-independent images.
- [x] `TI006` Align release and publish workflows, docs, and local developer entrypoints with the multi-image contract.
- [ ] `TI007` Run the validation ladder for the final changed-file set and retire or record any remaining divergence.

## Current Loop

- Date: 2026-03-25
- Current task: `TI006` complete; router publish or release flows now emit CPU and ROCm role images alongside the existing runtime images, and the next loop should focus on the final validation ladder in `TI007`
- Branch: `codex/topology-separation-research`
- Planned loop order:
  - `L1` document the current single-image payload and the compatibility-first rollout contract
  - `L2` add per-role image selection with no behavior change by default
  - `L3` carve out Envoy and dashboard images while preserving local startup and OpenClaw behavior
  - `L4` carve out router and ROCm images, then shrink the shared-image compatibility path
  - `L5` align CI or release flows, rerun validation, and record residual debt if needed
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/{README.md,governance.md,plans/README.md,tech-debt-register.md,tech-debt/README.md,architecture-guardrails.md}`, and `.agents/skills/harness/SKILL.md`
  - broad repository retrieval with `rg -n "VLLM_SR_IMAGE|vllm-sr-rocm|dashboard:|docker-publish|docker-release|start-envoy|start-router|start-dashboard|OPENCLAW|docker.io|podman|TARGET_ENVOY_URL|TARGET_ROUTER_URL|TARGET_ROUTER_METRICS_URL" src/vllm-sr dashboard tools/make .github/workflows docs/agent/plans -g '!**/.venv-codex/**'` because the dedicated `codebase-retrieval` tool was not available in this session
  - `make agent-report ENV=cpu CHANGED_FILES="docs/agent/plans/README.md tools/agent/repo-manifest.yaml docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md docs/agent/plans/pl-0015-local-runtime-topology-separation.md src/vllm-sr/Dockerfile src/vllm-sr/Dockerfile.rocm dashboard/backend/Dockerfile src/vllm-sr/cli/docker_start.py tools/make/docker.mk .github/workflows/docker-publish.yml .github/workflows/docker-release.yml src/vllm-sr/start-router.sh src/vllm-sr/start-envoy.sh src/vllm-sr/start-dashboard.sh"`
  - focused file reads for `src/vllm-sr/Dockerfile`, `src/vllm-sr/Dockerfile.rocm`, `dashboard/backend/Dockerfile`, `src/vllm-sr/start-{router,envoy,dashboard}.sh`, `src/vllm-sr/cli/docker_start.py`, `tools/make/docker.mk`, `.github/workflows/docker-publish.yml`, and `.github/workflows/docker-release.yml`
  - `make agent-validate`
  - `make agent-ci-gate CHANGED_FILES="docs/agent/plans/README.md docs/agent/plans/pl-0015-local-runtime-topology-separation.md docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md tools/agent/repo-manifest.yaml"` (blocked by pre-existing markdown lint failures in `e2e/config/models/mom-embedding-ultra/README.md`)
  - `make agent-report ENV=cpu CHANGED_FILES="src/vllm-sr/cli/docker_images.py src/vllm-sr/cli/docker_start.py src/vllm-sr/cli/commands/runtime.py tools/make/docker.mk e2e/testing/vllm-sr-cli/cli_test_base.py e2e/testing/vllm-sr-cli/run_cli_tests.py e2e/testing/vllm-sr-cli/test_unit_runtime_topology.py docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md"`
  - focused file reads for `src/vllm-sr/cli/{docker_images.py,docker_start.py,core.py,docker_backend.py,commands/runtime.py}`, `tools/make/docker.mk`, `e2e/testing/vllm-sr-cli/{cli_test_base.py,run_cli_tests.py,test_unit_runtime_topology.py,test_unit_serve.py,test_integration.py}`, `src/vllm-sr/tests/{test_openclaw_shared_network.py,test_deployment_backend.py,test_cli_main.py}`, `e2e/testing/run_memory_integration.sh`, `src/vllm-sr/rebuild-and-test.sh`, and `src/vllm-sr/Makefile`
  - `python3 -m pytest tests/test_docker_images.py tests/test_deployment_backend.py tests/test_openclaw_shared_network.py tests/test_cli_main.py -q` from `src/vllm-sr`
  - `bash -n e2e/testing/run_memory_integration.sh && bash -n src/vllm-sr/rebuild-and-test.sh`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" VLLM_SR_ROUTER_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_ENVOY_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_DASHBOARD_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest make vllm-sr-test`
  - `cd e2e/testing/vllm-sr-cli && PATH="$(git rev-parse --show-toplevel)/.venv-codex/bin:$PATH" CONTAINER_RUNTIME=docker VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_ROUTER_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_ENVOY_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_DASHBOARD_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_SIM_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-sim:latest RUN_INTEGRATION_TESTS=true python run_cli_tests.py --verbose --integration`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" VLLM_SR_ROUTER_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_ENVOY_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_DASHBOARD_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest make agent-serve-local ENV=cpu AGENT_STACK_NAME=role-image-contract AGENT_PORT_OFFSET=500`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-smoke-local AGENT_STACK_NAME=role-image-contract AGENT_PORT_OFFSET=500`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" VLLM_SR_STACK_NAME=role-image-contract VLLM_SR_PORT_OFFSET=500 vllm-sr stop`
  - `command -v kind && kind version` (failed because `kind` is not installed on this workstation, so the `ai-gateway` E2E selected by the harness remains environment-blocked)
  - `make agent-report ENV=cpu CHANGED_FILES="src/vllm-sr/Dockerfile.envoy src/vllm-sr/cli/consts.py src/vllm-sr/cli/docker_images.py src/vllm-sr/tests/test_docker_images.py tools/make/docker.mk .github/workflows/docker-publish.yml .github/workflows/docker-release.yml src/vllm-sr/Makefile src/vllm-sr/rebuild-and-test.sh src/vllm-sr/tests/test_makefile_surface.py docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md"`
  - focused file reads for `src/vllm-sr/{Dockerfile,Dockerfile.rocm,Dockerfile.envoy,start-envoy.sh,Makefile,rebuild-and-test.sh}`, `src/vllm-sr/cli/{consts.py,docker_images.py,core.py}`, `.github/workflows/{docker-publish.yml,docker-release.yml}`, and `tools/make/docker.mk`
  - `python3 -m pytest src/vllm-sr/tests/test_docker_images.py src/vllm-sr/tests/test_deployment_backend.py src/vllm-sr/tests/test_openclaw_shared_network.py src/vllm-sr/tests/test_cli_main.py src/vllm-sr/tests/test_makefile_surface.py -q`
  - `bash -n src/vllm-sr/rebuild-and-test.sh`
  - `python3 - <<'PY' ... yaml.safe_load('.github/workflows/docker-{publish,release}.yml') ... PY`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make vllm-sr-test`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-dev ENV=cpu` (intentionally canceled after confirming it was rebuilding unrelated monolithic `torch`/CUDA-heavy layers that were already present locally; targeted validation used the existing local `vllm-sr` image plus the newly built Envoy image instead)
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make vllm-sr-envoy-build`
  - `cd e2e/testing/vllm-sr-cli && PATH="$(git rev-parse --show-toplevel)/.venv-codex/bin:$PATH" CONTAINER_RUNTIME=docker VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_ROUTER_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_ENVOY_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-envoy:latest VLLM_SR_DASHBOARD_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_SIM_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-sim:latest RUN_INTEGRATION_TESTS=true python run_cli_tests.py --verbose --integration`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-serve-local ENV=cpu AGENT_STACK_NAME=envoy-image-smoke AGENT_PORT_OFFSET=600`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-smoke-local AGENT_STACK_NAME=envoy-image-smoke AGENT_PORT_OFFSET=600`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-stop-local AGENT_STACK_NAME=envoy-image-smoke AGENT_PORT_OFFSET=600`
  - `make agent-report ENV=cpu CHANGED_FILES="dashboard/backend/Dockerfile src/vllm-sr/cli/consts.py src/vllm-sr/cli/docker_images.py src/vllm-sr/tests/test_docker_images.py tools/make/docker.mk src/vllm-sr/Makefile src/vllm-sr/rebuild-and-test.sh src/vllm-sr/tests/test_makefile_surface.py e2e/testing/vllm-sr-cli/test_integration.py docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md"`
  - focused file reads for `dashboard/backend/Dockerfile`, `src/vllm-sr/cli/{consts.py,docker_images.py,docker_start.py,core.py}`, `tools/make/docker.mk`, `src/vllm-sr/{Makefile,rebuild-and-test.sh}`, and `e2e/testing/vllm-sr-cli/test_integration.py`
  - `cd src/vllm-sr && python3 -m pytest tests/test_docker_images.py tests/test_makefile_surface.py tests/test_cli_main.py tests/test_openclaw_shared_network.py tests/test_deployment_backend.py -q`
  - `make dashboard-check`
  - `bash -n src/vllm-sr/rebuild-and-test.sh`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make vllm-sr-test`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make vllm-sr-dashboard-build`
  - `cd e2e/testing/vllm-sr-cli && PATH="$(git rev-parse --show-toplevel)/.venv-codex/bin:$PATH" CONTAINER_RUNTIME=docker VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_ROUTER_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest VLLM_SR_ENVOY_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-envoy:latest VLLM_SR_DASHBOARD_IMAGE=ghcr.io/vllm-project/semantic-router/dashboard:latest VLLM_SR_SIM_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-sim:latest RUN_INTEGRATION_TESTS=true python run_cli_tests.py --verbose --integration`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-validate`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-serve-local ENV=cpu AGENT_STACK_NAME=dashboard-image-smoke AGENT_PORT_OFFSET=700`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" VLLM_SR_STACK_NAME=dashboard-image-smoke VLLM_SR_PORT_OFFSET=700 vllm-sr status all`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-smoke-local AGENT_STACK_NAME=dashboard-image-smoke AGENT_PORT_OFFSET=700`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make agent-stop-local AGENT_STACK_NAME=dashboard-image-smoke AGENT_PORT_OFFSET=700`
  - `make agent-e2e-affected CHANGED_FILES="dashboard/backend/Dockerfile,src/vllm-sr/cli/consts.py,src/vllm-sr/cli/docker_images.py,src/vllm-sr/tests/test_docker_images.py,tools/make/docker.mk,src/vllm-sr/Makefile,src/vllm-sr/rebuild-and-test.sh,src/vllm-sr/tests/test_makefile_surface.py,e2e/testing/vllm-sr-cli/test_integration.py,docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md"` (selects `ai-gateway` and fails locally because `kind` is not installed on this workstation)
  - `make agent-report ENV=cpu CHANGED_FILES="src/vllm-sr/Dockerfile.router src/vllm-sr/Dockerfile.router.rocm src/vllm-sr/cli/consts.py src/vllm-sr/cli/docker_images.py src/vllm-sr/Makefile src/vllm-sr/rebuild-and-test.sh src/vllm-sr/tests/test_docker_images.py src/vllm-sr/tests/test_makefile_surface.py src/vllm-sr/tests/test_openclaw_shared_network.py tools/make/docker.mk docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md"`
  - focused file reads for `src/vllm-sr/{Dockerfile,Dockerfile.rocm,Dockerfile.router,Dockerfile.router.rocm,start-router.sh,Makefile,rebuild-and-test.sh}`, `src/vllm-sr/cli/{consts.py,docker_images.py,core.py,runtime_stack.py}`, `tools/make/docker.mk`, and `src/vllm-sr/tests/{test_docker_images.py,test_makefile_surface.py,test_openclaw_shared_network.py,test_cli_main.py,test_deployment_backend.py}`
  - `cd src/vllm-sr && python3 -m pytest tests/test_openclaw_shared_network.py -k 'creates_and_connects_shared_network_without_observability or uses_state_root_override' -q`
  - `cd src/vllm-sr && python3 -m pytest tests/test_docker_images.py tests/test_makefile_surface.py tests/test_cli_main.py tests/test_deployment_backend.py tests/test_openclaw_shared_network.py -q`
  - `bash -n src/vllm-sr/rebuild-and-test.sh`
  - `PATH="$(pwd)/.venv-codex/bin:$PATH" make vllm-sr-test`
  - `make agent-validate`
  - `make agent-report ENV=cpu CHANGED_FILES=".github/workflows/docker-publish.yml .github/workflows/docker-release.yml tools/make/docker.mk src/vllm-sr/Makefile src/vllm-sr/rebuild-and-test.sh docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md"`
  - focused file reads for `.github/workflows/{docker-publish.yml,docker-release.yml}`, `src/vllm-sr/Makefile`, and `tools/make/docker.mk`
  - `python3 - <<'PY' ... yaml.safe_load('.github/workflows/docker-{publish,release}.yml') ... assert router jobs and matrices ... PY`
  - `make agent-validate`
  - `make agent-e2e-affected CHANGED_FILES=".github/workflows/docker-publish.yml .github/workflows/docker-release.yml docs/agent/plans/pl-0016-local-runtime-three-image-rollout.md"` (selected `ai-gateway` and then failed because `kind` is not installed on this workstation)
- Key findings:
  - The split runtime already starts three containers, but all three still pull from one image contract in CLI and Make.
  - The current `vllm-sr` and `vllm-sr-rocm` final images bundle router binaries, native libraries, Envoy, dashboard backend, frontend assets, Python CLI, OpenClaw assets, benchmark dependencies, model-eval dependencies, and `docker.io`, so dashboard and Envoy inherit substantial payload they do not own.
  - The existing standalone `dashboard/backend/Dockerfile` proves that a separately published dashboard image is operationally viable, but it does not yet carry the full local split-runtime payload needed for Docker-backed OpenClaw control and runtime helper flows.
  - Current release automation already publishes separate `dashboard`, `vllm-sr`, `vllm-sr-rocm`, and `vllm-sr-sim` images, which lowers the operational risk of extending the matrix for role-specific runtime images.
  - `src/vllm-sr/start-router.sh`, `src/vllm-sr/start-envoy.sh`, and `src/vllm-sr/start-dashboard.sh` already express role-specific entrypoints, so the main remaining work is packaging, image selection, and validation rather than inventing new container roles.
  - The safest rollout is compatibility-first: introduce per-role image settings that default to `VLLM_SR_IMAGE`, then carve out role images one at a time instead of replacing the monolithic image in one jump.
  - `make agent-validate` passes for this plan and index update; `make agent-ci-gate` currently fails for unrelated repository markdown lint violations in `e2e/config/models/mom-embedding-ultra/README.md`, not for the new plan content.
  - `TI002` now exposes explicit router, Envoy, and dashboard image contracts through CLI flags, host environment variables, and repo-native Make targets while preserving the original `--image` and `VLLM_SR_IMAGE` fallback behavior.
  - CLI argument precedence matters: explicit role flags must win first, explicit `--image` must win over environment-level role overrides, and environment-level role overrides should only shadow `VLLM_SR_IMAGE` or default selection. Without that ordering, `vllm-sr-test` and the integration runner can no longer validate missing-image behavior safely when role-image env vars are present.
  - Repo-native wrappers that previously hard-coded `serve --image ...` needed to move to `VLLM_SR_IMAGE=... serve ...`; otherwise the new role-image env contract was technically present but unreachable from common local-dev entrypoints.
  - CPU-local smoke now passes on a stack started with explicit `VLLM_SR_{ROUTER,ENVOY,DASHBOARD}_IMAGE` values, so the new contract survives real local startup, health checks, and stop behavior instead of only passing unit tests.
  - The harness still selects `ai-gateway` as the affected local E2E profile for this startup-chain change, but this workstation cannot run it because `kind` is absent from `PATH`; that remains an environment blocker rather than a failure in the new image-contract work.
  - `TI003` now publishes and builds a dedicated `ghcr.io/vllm-project/semantic-router/vllm-sr-envoy` image from `src/vllm-sr/Dockerfile.envoy`, and both direct `vllm-sr serve` and repo-native Make flows can select it without changing the router or dashboard image contract.
  - Direct CLI startup now derives the Envoy image from official `vllm-sr` or `vllm-sr-rocm` tags when the base image is official, so the canonical `vllm-sr serve --image-pull-policy never` path picks up `vllm-sr-envoy:latest` automatically instead of silently falling back to the monolithic image.
  - The first local smoke attempt surfaced a real operational gap: the Envoy image must retain `curl`, because `vllm-sr status` and `make agent-smoke-local` exec into the container and probe `http://localhost:9901/ready`. A service-only image that omits that probe dependency breaks the existing health-check contract even when Envoy itself is healthy.
  - The attempted `agent-dev` rebuild showed the practical payoff of extraction: the monolithic `vllm-sr` image still drags in `bench` and `model_eval` layers, including large `torch`/CUDA-related dependencies that Envoy does not own, while the dedicated Envoy image rebuilds in seconds once its small Python layer is cached.
  - `TI004` now gives dashboard its own runtime image contract and default derivation path, so direct `vllm-sr serve --image-pull-policy never` resolves official `ghcr.io/vllm-project/semantic-router/dashboard:<tag>` images instead of silently falling back to the monolithic runtime image when the base tag is official.
  - The dashboard role image needs more than the pre-existing standalone backend binary and frontend assets: local split-runtime flows also require Docker CLI access for OpenClaw, `/app/cli` plus Python dependencies for runtime helpers, `tools_db.json`, and the start script used by the shared runtime image.
  - Alpine's externally managed Python environment blocks the old in-image `pip install` pattern, so the dashboard role image now installs its Python helper dependencies in a dedicated virtual environment instead of mutating the system interpreter.
  - The first dashboard smoke attempt surfaced the same contract class as Envoy: `vllm-sr status` and `make agent-smoke-local` exec `curl` inside the dashboard container for `http://localhost:8700/healthz`, so trimming `curl` from the role image breaks health checks even when the dashboard service itself is healthy.
  - CLI integration needed one harness-side adjustment for the split-image world: a successful `vllm-sr serve` that exits quickly after launching the stack should be treated as valid when the runtime containers are up, rather than being misreported as a crash.
  - CPU-local serve, `vllm-sr status all`, smoke, and stop all pass with router, Envoy, and dashboard on independent image contracts (`vllm-sr`, `vllm-sr-envoy`, and `dashboard` respectively), so the dashboard extraction preserves the split-topology runtime contract instead of only passing unit tests.
  - The harness still selects `ai-gateway` as the affected local E2E profile for the combined startup-chain change set, and it still fails on this workstation because `kind` is missing from `PATH`; that remains an environment blocker rather than a dashboard-image regression.
  - `TI005` now adds dedicated router runtime images for both CPU and ROCm in `src/vllm-sr/Dockerfile.router` and `src/vllm-sr/Dockerfile.router.rocm`, so local split runtime no longer depends on the monolithic `vllm-sr` image when resolving official router role images.
  - Router image extraction can be much narrower than the monolithic runtime: the role image only needs the router binary, native libraries, `start-router.sh`, `bash`, `curl`, and a minimal Python plus `PyYAML` helper path for setup-mode config inspection.
  - The ROCm router role image still owns the ONNX Runtime ROCm wheel and flash-attention helper library path, but dashboard and Envoy do not, which keeps the ROCm-specific payload scoped to the role that actually executes model routing.
  - Default image derivation now maps official `ghcr.io/vllm-project/semantic-router/vllm-sr:*` tags to `vllm-sr-router:*` or `vllm-sr-router-rocm:*` for the router role, so canonical `vllm-sr serve --image-pull-policy never` keeps working without requiring users to opt into explicit router overrides first.
  - The only test regressions from `TI005` were stale OpenClaw shared-network fixtures that still assumed either a single legacy runtime container or real Docker cleanup, which confirms the production runtime contract stayed intact while the test harness assumptions were updated.
  - `TI006` exposed a genuine packaging mismatch: local CLI and Make already resolve `vllm-sr-router` and `vllm-sr-router-rocm`, but the GHCR publish and tagged release workflows were not producing those images yet, which meant the new defaults could not be relied on outside a developer machine with locally built images.
  - `docker-publish.yml` now treats `vllm-sr-router` as a first-class multi-arch runtime image beside `vllm-sr`, `vllm-sr-envoy`, `dashboard`, and `vllm-sr-sim`, and it also builds a dedicated `vllm-sr-router-rocm` amd64 artifact beside `vllm-sr-rocm`.
  - `docker-release.yml` now tags and pushes both `vllm-sr-router` and `vllm-sr-router-rocm` for versioned releases, which closes the gap between the local role-image contract and the published image catalog.
  - The remaining validation blocker after `TI006` is unchanged from earlier loops: the harness-selected `ai-gateway` local E2E still cannot run on this workstation because `kind` is not installed, so the unresolved risk is environmental rather than a newly introduced workflow or runtime mismatch.

## Decision Log

- 2026-03-25: Three-imageization is a follow-on packaging workstream, not part of the already-landed topology split; keep topology and packaging changes in separate execution plans.
- 2026-03-25: The rollout should start by adding explicit `router`, `envoy`, and `dashboard` image contracts that all fall back to the current `VLLM_SR_IMAGE`.
- 2026-03-25: The first extracted image should be Envoy because it has the smallest runtime surface and the lowest coupling to native router libraries.
- 2026-03-25: Dashboard image extraction should build on the existing standalone dashboard image path, but it must retain Docker-backed OpenClaw and local runtime helper support before it can replace the shared runtime image in local serve flows.
- 2026-03-25: Router image extraction should land after Envoy and dashboard, and ROCm support should remain router-specific unless later evidence shows a narrower dashboard or Envoy requirement.
- 2026-03-25: `TI002` will treat explicit CLI image flags as higher priority than environment-driven image overrides; service-specific env vars only override `VLLM_SR_IMAGE` and defaults, not an explicit `--image` chosen for a one-off command.
- 2026-03-25: Local wrappers that call `serve` should prefer setting `VLLM_SR_IMAGE` in the host environment over passing `--image` inline when they want downstream role-image env overrides to remain effective.
- 2026-03-25: `TI003` should keep `curl` inside the Envoy role image because the existing CLI status and smoke contracts exec probe commands inside the container; trimming that utility breaks behavior even though Envoy itself is healthy.
- 2026-03-25: `TI004` should install dashboard-side Python helper dependencies into a dedicated virtual environment inside the role image because Alpine's externally managed Python environment rejects direct system-level `pip install`.
- 2026-03-25: `TI004` should keep `curl` inside the dashboard role image because the existing CLI status and smoke contracts exec dashboard health probes from inside the container, and removing that utility breaks the contract even when the service is healthy.
- 2026-03-25: The CLI integration harness should treat an early, successful `vllm-sr serve` exit as valid when the target runtime containers are already running, because split-image startup can finish before the harness's old long-running-process assumption.
- 2026-03-25: `TI005` should derive official router role images automatically from official `vllm-sr` base tags so the canonical local serve flow picks up `vllm-sr-router` and `vllm-sr-router-rocm` without requiring separate user flags.
- 2026-03-25: `TI005` should keep `bash`, `curl`, and a minimal `python3` plus `PyYAML` path inside router role images because `start-router.sh`, setup-mode config inspection, and current in-container health or status probes all depend on them even after the router runtime payload is narrowed.
- 2026-03-25: `TI006` should publish `vllm-sr-router` as a multi-arch CPU image and `vllm-sr-router-rocm` as an amd64-only ROCm image so the official GHCR catalog matches the role-image defaults used by local CLI and Make flows.

## Follow-up Debt / ADR Links

- Predecessor execution plan:
  - [PL-0015 local runtime topology separation](pl-0015-local-runtime-topology-separation.md)
- Related debt to watch while executing this plan:
  - [TD004](../tech-debt/td-004-python-cli-kubernetes-workflow-separation.md)
  - [TD025](../tech-debt/td-025-dashboard-backend-runtime-control-slice-collapse.md)
  - [TD031](../tech-debt/td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
  - [TD034](../tech-debt/td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
