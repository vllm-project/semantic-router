# E2E Profile Selection Playbook

Use [tools/agent/e2e-profile-map.yaml](../../../tools/agent/e2e-profile-map.yaml) as the source of truth for durable E2E and integration-suite selection.

## Suite Taxonomy

- Standard profile E2E
  - Go profile-based suites under `e2e/profiles/**`
  - selected through `profile_rules`
  - targeted by `make agent-e2e-affected CHANGED_FILES="..."`
- Manual-only profile E2E
  - Go profile-based suites that remain outside the standard CI matrix
  - selected through `manual_profile_rules`
  - reported by `make agent-report`, but not auto-added to the standard CI profile set
- Workflow-driven integration suites
  - durable suites under `e2e/testing/**` or dedicated CI workflows
  - selected through `workflow_suite_rules`
  - reported with named local commands even when they do not map to a Go profile

## Standard Profiles

- `kubernetes`
  - shared baseline router contract
  - default local profile and one of the two steady-state shared suites
- `dashboard`
  - shared dashboard API contract
  - one of the two steady-state shared suites

## Manual-Only Profiles

- `aibrix`
  - AIBrix gateway/control-plane health plus smoke routing
- `routing-strategies`
  - keyword, entropy, and fallback routing semantics
- `dynamic-config`
  - CRD-driven routing and embedding-signal behavior
- `llm-d`
  - llm-d inference-gateway health plus smoke routing
- `istio`
  - service-mesh sidecar, mTLS, traffic, and tracing contracts
- `production-stack`
  - HA, load-balancing, failover, and throughput semantics
- `ml-model-selection`
  - ML model-selection behavior
- `multi-endpoint`
  - per-environment routing and safety policy behavior
- `authz-rbac`
  - authz-driven routing and rate limiting
- `streaming`
  - streamed request-body and streaming-cache behavior
- `response-api`
  - memory-backed Responses API behavior, currently kept outside the standard CI matrix
- `response-api-redis`
  - Redis-backed Responses API behavior and TTL expiry
- `response-api-redis-cluster`
  - Redis Cluster-backed Responses API behavior and TTL expiry
- `dynamo`
  - GPU- and runtime-specific Dynamo coverage
- `rag-hybrid-search`
  - Llama Stack-backed RAG vector-store and hybrid-search coverage

## Workflow-Driven Integration Suites

- `docker-shared-integration`
  - workflow: [integration-test-vllm-sr-cli.yml](../../../.github/workflows/integration-test-vllm-sr-cli.yml)
  - local command: `make vllm-sr-test-integration VLLM_SR_TEST_TARGET=docker`
- `k8s-shared-integration`
  - workflow: [integration-test-k8s.yml](../../../.github/workflows/integration-test-k8s.yml)
  - local command: `make vllm-sr-test-integration VLLM_SR_TEST_TARGET=k8s`
- `memory-integration`
  - workflow: [integration-test-memory.yml](../../../.github/workflows/integration-test-memory.yml)
  - local command: `make memory-test-integration`
  - retained as a manual/nightly exception, not part of the steady-state shared-suite PR path

## Selection Rules

- Common E2E framework changes trigger:
  - explicit local default profile: `kubernetes`
  - the shared docker and k8s workflow-driven suites for `kubernetes` and `dashboard`
- Standard profile-local changes trigger only the matching local and CI profiles.
- Manual-only profile changes trigger the matching local profile and stay outside the standard CI profile list.
- Workflow-driven integration changes trigger the named workflow suite command from `workflow_suite_rules`.
- `make agent-feature-gate` does not auto-run local E2E.
- `make agent-e2e-affected CHANGED_FILES="..."` remains the explicit local command when affected profiles need to be run by hand.
- If a durable suite remains outside the standard profile matrix, document it in `workflow_suite_rules` or `manual_profile_rules` instead of leaving it as an implicit CI-only path.

## Canonical Commands

- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-e2e-affected CHANGED_FILES="..."`
- `make e2e-test E2E_PROFILE=<profile>`
- `make vllm-sr-test-integration VLLM_SR_TEST_TARGET=docker`
- `make vllm-sr-test-integration VLLM_SR_TEST_TARGET=k8s`
- `make memory-test-integration`
