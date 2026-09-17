# Repository tooling tests that do not require a running Router.

test-calibration: $(HARNESS_VENV_DEPS) ## Test recipe probes and offline tuning tools
	@PYTHONPATH="$(CURDIR)/tools/calibration:$(CURDIR)/tools/calibration/recipe$${PYTHONPATH:+:$${PYTHONPATH}}" \
		"$(AGENT_PYTHON)" -m pytest tools/calibration/recipe tools/calibration/tuning/tests

PROVIDER_SIMULATOR_VENV ?= $(CURDIR)/.agent-harness/provider-simulator
PROVIDER_SIMULATOR_PYTHON := $(PROVIDER_SIMULATOR_VENV)/bin/python

# Keep service dependencies separate from the harness and CLI environments.
$(PROVIDER_SIMULATOR_VENV)/.requirements: tools/test/services/mock-vllm/requirements.txt tools/test/services/mock-vllm/requirements-dev.txt | $(HARNESS_VENV_DEPS)
	@"$(AGENT_PYTHON)" -m venv "$(PROVIDER_SIMULATOR_VENV)"
	@"$(PROVIDER_SIMULATOR_PYTHON)" -m pip install -r tools/test/services/mock-vllm/requirements-dev.txt
	@touch "$@"

test-provider-simulator: $(PROVIDER_SIMULATOR_VENV)/.requirements ## Test the provider simulator's protocol contracts
	@cd tools/test/services/mock-vllm && "$(PROVIDER_SIMULATOR_PYTHON)" -m pytest tests

.PHONY: test-calibration test-provider-simulator
