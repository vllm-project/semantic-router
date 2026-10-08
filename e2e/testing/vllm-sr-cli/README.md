# vLLM Semantic Router CLI tests

This suite checks the `vllm-sr` command surface and its local container
topology. Unit tests inspect generated commands without starting services;
integration tests start the split Router, Envoy, and dashboard
images (Envoy with `--gateway extproc`; the standalone module runs without it)
and exercise live APIs.

## Run through Make

From the repository root:

```bash
make vllm-sr-test
make vllm-sr-test-integration
```

`vllm-sr-test` bootstraps the editable CLI and runs without a container
daemon. `vllm-sr-test-integration` builds the required local images and needs
Docker or Podman access.

## What the suite covers

| File | Scope |
| --- | --- |
| `test_unit_serve.py` | Config bootstrap, mounts, ports, image pull policy, tokens, and read-only mode. |
| `test_unit_lifecycle.py` | `status`, `logs`, `stop`, `dashboard`, and `config` command construction. |
| `test_unit_runtime_topology.py` | Split-runtime discovery, cleanup, timeouts, and Docker/Podman selection. |
| `test_integration.py` | Live health, management APIs, model visibility, path rewrites, sidecars, lifecycle, and pull policies. |
| `test_integration_storage_isolation.py` | Redis and Postgres answer on the stack's data network and are unreachable from its application network. |
| `test_integration_standalone.py` | Standalone mode on the docker target: no Envoy container, and a chat request routed through the Router's own listener. |
| `test_integration_first_run_setup.py` | `vllm-sr serve` in an empty directory opens setup and waits; activation through the Dashboard's API makes the CLI create the Router from the activated config. The Dashboard reports the Router in standby, then running, without a container runtime. |
| `test_integration_recipe_bearer.py` | The Dashboard imports a Recipe with bearer management auth (`fixtures/bearer-recipe`) from a local HTTPS server signed by a test-only CA, and `vllm-sr serve`, run as the suite's user (non-root in CI), activates it: the Router accepts only the management credential the CLI generated, and no file but the CLI's owner-only state holds it. |
| `test_integration_model_runtime.py` | The Quickstart's decision signal on a `model_runtime` deployment the Router starts and asks inside its container. |
| `test_integration_engine_mode.py` | `vllm-sr serve <model>` runs the model runtime in a container from the router image, answers the Quickstart's requests on the published port, and stops and removes itself on Ctrl-C. |
| `test_integration_plugin_example.py` | The plugin guide's "Try it": pip installs the example plugin, serves its keyword package and gets the guide's answer, also on the example's own accelerator and profile, and from the guide's image with `vllm-sr serve`. |
| `cli_test_base.py` | Shared command and container helpers, including the check that the Dashboard mounts no runtime socket and has no container CLI, which every serve module runs. |
| `serve_session.py` | Background `vllm-sr serve` orchestration shared by the integration modules. |
| `mock_upstream.py` | The mock OpenAI upstream that integration modules send chat requests to. |
| `runtime_http.py` | A model runtime started on a free port, the JSON calls sent to it, the requests a docs page shows, and fixture packages written by the router image's runtime. |
| `run_cli_tests.py` | Prerequisite checks, discovery, filtering, and reporting. |

The test files are the source of truth for individual assertions; this README
describes stable areas instead of duplicating every test name.

## Run the test runner directly

Install the editable CLI first, then run from this directory:

```bash
python run_cli_tests.py --verbose
python run_cli_tests.py --verbose --integration
python run_cli_tests.py --pattern lifecycle
```

Set `CONTAINER_RUNTIME=docker` or `CONTAINER_RUNTIME=podman` to select a
runtime. `RUN_INTEGRATION_TESTS=true` also enables integration discovery, but
the `--integration` flag is clearer for direct runs. The Make target supplies
the local image names used by the full integration suite.
