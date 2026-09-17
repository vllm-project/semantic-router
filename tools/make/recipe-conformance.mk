# ======== recipe-conformance.mk ========
# = Maintained recipe conformance gates =
# =======================================

RECIPE_CONFORMANCE_PYTHON ?= $(if $(wildcard $(CURDIR)/.venv-agent/bin/python),$(CURDIR)/.venv-agent/bin/python,python3)
RECIPE_CONFORMANCE_REPORT_DIR ?= $(CURDIR)/.agent-harness/recipe-conformance
RECIPE_CONFORMANCE_SHARDS ?= 3
RECIPE_CONFORMANCE_RECIPE ?=
VLLM_SR_PORT_OFFSET ?= 0
RECIPE_CONFORMANCE_ROUTER_URL ?= http://127.0.0.1:$(shell expr 8080 + $(VLLM_SR_PORT_OFFSET))
RECIPE_CONFORMANCE_RECIPES_ROOT ?= $(CURDIR)/config/recipes
RECIPE_CONFORMANCE_RECIPES ?=

##@ Recipe Conformance

recipe-conformance-assets: vllm-sr-install-cli ## Validate maintained assets and source CLI recipe composition
	@$(LOG_TARGET)
	@$(RECIPE_CONFORMANCE_PYTHON) -m unittest \
		tools/calibration/recipe/router_calibration_fixture_test.py \
		tools/calibration/recipe/router_calibration_support_test.py \
		tools/calibration/recipe/router_calibration_signal_values_test.py \
		tools/calibration/recipe/recipe_conformance_test.py \
		tools/calibration/recipe/recipe_conformance_built_in_test.py
	@$(RECIPE_CONFORMANCE_PYTHON) tools/calibration/recipe/recipe_conformance.py \
		--output-dir "$(RECIPE_CONFORMANCE_REPORT_DIR)" \
		static-all
recipe-conformance-static: recipe-conformance-assets ## Validate assets and Router composition tests
	@cd src/semantic-router && go test \
		./pkg/config/... \
		./pkg/dsl/... \
		./pkg/decision/...

recipe-conformance-plan: ## Emit deterministic live-CPU recipe shards
	@$(LOG_TARGET)
	@$(RECIPE_CONFORMANCE_PYTHON) tools/calibration/recipe/recipe_conformance.py \
		plan-all --shards "$(RECIPE_CONFORMANCE_SHARDS)"

recipe-conformance-report: ## Assemble downloaded shard artifacts into one report
	@$(LOG_TARGET)
	@$(RECIPE_CONFORMANCE_PYTHON) tools/calibration/recipe/recipe_conformance.py \
		--output-dir "$(RECIPE_CONFORMANCE_REPORT_DIR)" \
		report-all

recipe-conformance-eval: ## Evaluate one active recipe router (set RECIPE_CONFORMANCE_RECIPE)
	@$(LOG_TARGET)
	@if [ -z "$(RECIPE_CONFORMANCE_RECIPE)" ]; then \
		echo "RECIPE_CONFORMANCE_RECIPE is required"; \
		exit 2; \
	fi
	@$(RECIPE_CONFORMANCE_PYTHON) tools/calibration/recipe/recipe_conformance.py \
		--output-dir "$(RECIPE_CONFORMANCE_REPORT_DIR)" \
		--recipes-root "$(RECIPE_CONFORMANCE_RECIPES_ROOT)" \
		eval \
		--recipe "$(RECIPE_CONFORMANCE_RECIPE)" \
		--router-url "$(RECIPE_CONFORMANCE_ROUTER_URL)"

recipe-conformance-live-cpu: ## Build once and run live CPU probes (set RECIPE_CONFORMANCE_RECIPES)
	@$(LOG_TARGET)
	@if [ -z "$(RECIPE_CONFORMANCE_RECIPES)" ]; then \
		echo "RECIPE_CONFORMANCE_RECIPES is required"; \
		exit 2; \
	fi
	@$(MAKE) vllm-sr-router-build vllm-sr-envoy-build
	@RECIPES="$(RECIPE_CONFORMANCE_RECIPES)" \
		RECIPES_ROOT="$(RECIPE_CONFORMANCE_RECIPES_ROOT)" \
		ROUTER_IMAGE="$(VLLM_SR_ROUTER_IMAGE)" \
		VLLM_SR_ENVOY_IMAGE="$(VLLM_SR_ENVOY_IMAGE)" \
		ROUTER_URL="$(RECIPE_CONFORMANCE_ROUTER_URL)" \
		REPORT_ROOT="$(RECIPE_CONFORMANCE_REPORT_DIR)" \
		bash e2e/testing/run_recipe_conformance.sh

recipe-conformance-live-cpu-all: ## Build once and run every CPU-compatible live source
	@$(MAKE) vllm-sr-router-build vllm-sr-envoy-build
	@set -e; sources="$$(mktemp)"; trap 'rm -f "$$sources"' EXIT; \
	$(RECIPE_CONFORMANCE_PYTHON) tools/calibration/recipe/recipe_conformance.py sources --format pipe > "$$sources"; \
	while IFS='|' read -r source recipes_root report_dir recipes; do \
		[ -n "$$recipes" ] || continue; \
		RECIPES="$$recipes" RECIPES_ROOT="$$recipes_root" \
		ROUTER_IMAGE="$(VLLM_SR_ROUTER_IMAGE)" \
		VLLM_SR_ENVOY_IMAGE="$(VLLM_SR_ENVOY_IMAGE)" \
		ROUTER_URL="$(RECIPE_CONFORMANCE_ROUTER_URL)" \
		REPORT_ROOT="$(RECIPE_CONFORMANCE_REPORT_DIR)/$$report_dir" \
		bash e2e/testing/run_recipe_conformance.sh; \
	done < "$$sources"

.PHONY: recipe-conformance-assets recipe-conformance-static recipe-conformance-plan \
	recipe-conformance-report \
	recipe-conformance-eval recipe-conformance-live-cpu \
	recipe-conformance-live-cpu-all
