# vLLM-SR CLI Tests

End-to-end tests for the `vllm-sr` command-line interface.

## Quick Start

```bash
# From project root:
make vllm-sr-test                                                 # Unit tests only (fast)
make vllm-sr-test-integration VLLM_SR_TEST_TARGET=docker          # Shared suites against docker
make vllm-sr-test-integration VLLM_SR_TEST_TARGET=k8s             # Shared suites against managed Kind
```

## Make Targets

| Target | Description | Requires |
|--------|-------------|----------|
| `make vllm-sr-test` | Run unit tests only | Python (bootstraps local `vllm-sr` CLI) |
| `make vllm-sr-test-integration VLLM_SR_TEST_TARGET=docker` | Run shared suites against docker | Local images (build automatically) |
| `make vllm-sr-test-integration VLLM_SR_TEST_TARGET=k8s` | Run shared suites against managed Kind | Local images plus `kind`, `kubectl`, and `helm` |

## Test Files

| File | Type | Description |
|------|------|-------------|
| `test_unit_serve.py` | Unit | Tests `serve` flags |
| `test_unit_lifecycle.py` | Unit | Tests `status/logs/stop/dashboard/config` flags |
| `test_integration.py` | **Integration** | Real container tests (strong validation) |
| `test_shared_suites.py` | **Integration** | Shared `kubernetes` and `dashboard` contracts against docker or k8s |
| `cli_test_base.py` | Helper | Base class with utilities |
| `run_cli_tests.py` | Helper | Test runner |

## Integration Tests (Strong Validation)

These tests start real docker or k8s environments through `vllm-sr` and verify the default shared contracts:

| Test | What it verifies |
|------|------------------|
| `test_running_container_contracts` | canonical config → serve → router container running → health |
| `test_fleet_sim_sidecar_contracts` | `vllm-sr serve` starts the simulator sidecar and exposes `/healthz` |
| `test_env_var_passed_to_container` | HF_TOKEN inside container |
| `test_volume_mounting` | config.yaml + models/ mounted |
| `test_status_shows_running_container` | `status` reports running |
| `test_logs_retrieves_container_logs` | `logs` gets actual output |
| `test_stop_terminates_container` | `stop` actually stops container |
| `test_image_pull_policy_never_fails_with_missing_image` | `never` policy rejects missing image |
| `test_image_pull_policy_always_attempts_pull` | `always` policy attempts pull |
| `test_chat_completions_and_models_contract` | shared `kubernetes` contract across docker and k8s |
| `test_dashboard_health_status_and_config_contract` | shared `dashboard` contract across docker and k8s |

## Unit Tests (Flag Validation)

| Command | Options Tested |
|---------|----------------|
| `serve` | `--config`, `--image`, `--image-pull-policy`, `--readonly-dashboard` |
| `status` | `all`, `envoy`, `router`, `dashboard`, `simulator` |
| `logs` | `envoy`, `router`, `dashboard`, `simulator`, `-f/--follow` |
| `stop` | default |
| `dashboard` | default, `--no-open` |
| `config` | `envoy`, `router` |

## Running Tests

```bash
cd e2e/testing/vllm-sr-cli

# All unit tests
python run_cli_tests.py --verbose

# Include integration tests against docker
VLLM_SR_TEST_TARGET=docker python run_cli_tests.py --verbose --integration

# Include integration tests against managed Kind
VLLM_SR_TEST_TARGET=k8s python run_cli_tests.py --verbose --integration

# Filter by pattern
python run_cli_tests.py --pattern lifecycle
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUN_INTEGRATION_TESTS` | Set to `true` to enable integration tests |
| `CONTAINER_RUNTIME` | Override runtime (`docker` only; `podman` is rejected) |
| `VLLM_SR_TEST_TARGET` | Shared integration target (`docker` or `k8s`) |
| `VLLM_SR_SHARED_SUITE` | Optional shared suite selector (`kubernetes` or `dashboard`) |
