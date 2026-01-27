# vLLM-SR CLI Tests

End-to-end tests for the `vllm-sr` command-line interface.

## Quick Start

```bash
# From project root:
make vllm-sr-test              # Unit tests only (fast)
make vllm-sr-test-integration  # Unit + Integration tests
```

## Make Targets

| Target | Description | Requires |
|--------|-------------|----------|
| `make vllm-sr-test` | Run unit tests only | Python, vllm-sr CLI |
| `make vllm-sr-test-integration` | Run unit + integration tests | Docker image (builds automatically) |

## Test Files

| File | Type | Description |
|------|------|-------------|
| `test_unit_init.py` | Unit | Tests `init` command |
| `test_unit_serve.py` | Unit | Tests `serve` flags |
| `test_unit_lifecycle.py` | Unit | Tests `status/logs/stop/dashboard/config` flags |
| `test_integration.py` | **Integration** | Real container tests (strong validation) |
| `cli_test_base.py` | Helper | Base class with utilities |
| `run_cli_tests.py` | Helper | Test runner |

## Integration Tests (Strong Validation)

These tests start real containers and verify with `docker inspect`:

| Test | What it verifies |
|------|------------------|
| `test_serve_full_startup` | init → serve → container running → health |
| `test_env_var_passed_to_container` | HF_TOKEN inside container |
| `test_volume_mounting` | config.yaml + models/ mounted |
| `test_status_shows_running_container` | `status` reports running |
| `test_logs_retrieves_container_logs` | `logs` gets actual output |
| `test_stop_terminates_container` | `stop` actually stops container |
| `test_image_pull_policy_never_fails_with_missing_image` | `never` policy rejects missing image |
| `test_image_pull_policy_always_attempts_pull` | `always` policy attempts pull |

## Unit Tests (Flag Validation)

| Command | Options Tested |
|---------|----------------|
| `init` | default, `--force` |
| `serve` | `--config`, `--image`, `--image-pull-policy`, `--readonly-dashboard` |
| `status` | `all`, `envoy`, `router`, `dashboard` |
| `logs` | `envoy`, `router`, `dashboard`, `-f/--follow` |
| `stop` | default |
| `dashboard` | default, `--no-open` |
| `config` | `envoy`, `router` |

## Running Tests

```bash
cd e2e/testing/vllm-sr-cli

# All unit tests
python run_cli_tests.py --verbose

# Include integration tests
python run_cli_tests.py --verbose --integration

# Filter by pattern
python run_cli_tests.py --pattern init
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUN_INTEGRATION_TESTS` | Set to `true` to enable integration tests |
| `CONTAINER_RUNTIME` | Override runtime (`docker` or `podman`) |
