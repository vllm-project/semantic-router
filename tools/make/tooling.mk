# Repository tooling tests that do not require a running Router.

test-calibration: $(HARNESS_VENV_DEPS) ## Test recipe probes and offline tuning tools
	@PYTHONPATH="$(CURDIR)/tools/calibration:$(CURDIR)/tools/calibration/recipe$${PYTHONPATH:+:$${PYTHONPATH}}" \
		"$(AGENT_PYTHON)" -m pytest tools/calibration/recipe tools/calibration/tuning/tests

PROVIDER_MOCKER_VENV ?= $(CURDIR)/.agent-harness/provider-mocker
PROVIDER_MOCKER_PYTHON := $(PROVIDER_MOCKER_VENV)/bin/python
PROVIDER_MOCKER_BOOTSTRAP_PYTHON ?= $(AGENT_PYTHON)

# Keep service dependencies separate from the harness and CLI environments.
$(PROVIDER_MOCKER_VENV)/.requirements: tools/test/services/provider-mocker/requirements.txt tools/test/services/provider-mocker/requirements-dev.txt | $(HARNESS_VENV_DEPS)
	@"$(PROVIDER_MOCKER_BOOTSTRAP_PYTHON)" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "provider-mocker requires Python 3.11+; set PROVIDER_MOCKER_BOOTSTRAP_PYTHON")'
	@"$(PROVIDER_MOCKER_BOOTSTRAP_PYTHON)" -m venv "$(PROVIDER_MOCKER_VENV)"
	@"$(PROVIDER_MOCKER_PYTHON)" -m pip install -r tools/test/services/provider-mocker/requirements-dev.txt
	@touch "$@"

test-provider-mocker: $(PROVIDER_MOCKER_VENV)/.requirements ## Test the provider mocker's protocol contracts
	@cd tools/test/services/provider-mocker && "$(PROVIDER_MOCKER_PYTHON)" -m pytest tests

.PHONY: test-calibration test-provider-mocker

TINY_MODEL_PORT ?= 8000

tiny-model-serve: ## Serve the pinned Qwen3-0.6B with the upstream CPU image
	@CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) python3 tools/test/services/tiny-model/run.py serve --port $(TINY_MODEL_PORT)

tiny-model-smoke: ## Download the pinned tiny model and check real generation, SSE and stopping
	@CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) python3 tools/test/services/tiny-model/run.py smoke --port 0

test-tiny-model: ## Test real-model smoke acceptance criteria without downloading weights
	@python3 -m unittest discover -s tools/test/services/tiny-model -p 'test_*.py'

.PHONY: tiny-model-serve tiny-model-smoke test-tiny-model

provider-mocker-install: $(PROVIDER_MOCKER_VENV)/.requirements ## Install the isolated provider mocker test environment

start-provider-mocker: provider-mocker-install ## Run the deterministic test backend in the foreground
	@cd tools/test/services/provider-mocker && PROVIDER_MOCKER_MODEL="$(PROVIDER_MOCKER_MODEL)" PROVIDER_MOCKER_SCENARIO="$(PROVIDER_MOCKER_SCENARIO)" "$(PROVIDER_MOCKER_PYTHON)" -m provider_mocker --host 127.0.0.1 --port $(PROVIDER_MOCKER_PORT)

.PHONY: provider-mocker-install start-provider-mocker
