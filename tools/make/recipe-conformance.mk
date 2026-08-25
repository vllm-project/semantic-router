# ======== recipe-conformance.mk ========
# = Maintained recipe conformance gates =
# =======================================

RECIPE_CONFORMANCE_PYTHON ?= $(if $(wildcard $(CURDIR)/.venv-agent/bin/python),$(CURDIR)/.venv-agent/bin/python,python3)
RECIPE_CONFORMANCE_REPORT_DIR ?= $(CURDIR)/.agent-harness/recipe-conformance
RECIPE_CONFORMANCE_SHARDS ?= 3
RECIPE_CONFORMANCE_RECIPE ?=
RECIPE_CONFORMANCE_ROUTER_URL ?= http://127.0.0.1:8080
RECIPE_CONFORMANCE_ENTRYPOINTS ?=
RECIPE_CONFORMANCE_RECIPES ?=

##@ Recipe Conformance

recipe-conformance-static: ## Validate all maintained recipe assets and probe contracts
	@$(LOG_TARGET)
	@$(RECIPE_CONFORMANCE_PYTHON) -m unittest \
		tools/agent/scripts/router_calibration_fixture_test.py \
		tools/agent/scripts/router_calibration_support_test.py \
		tools/agent/scripts/recipe_conformance_test.py
	@$(RECIPE_CONFORMANCE_PYTHON) tools/agent/scripts/recipe_conformance.py \
		--output-dir "$(RECIPE_CONFORMANCE_REPORT_DIR)" \
		static
	@$(RECIPE_CONFORMANCE_PYTHON) tools/agent/scripts/recipe_conformance.py \
		--recipes-root "$(CURDIR)/config/recipes/built-in/latest" \
		--output-dir "$(RECIPE_CONFORMANCE_REPORT_DIR)/built-in/latest" \
		--skip-root-readme \
		static
	@cd src/semantic-router && go test \
		./pkg/config/... \
		./pkg/dsl/... \
		./pkg/decision/...

recipe-conformance-plan: ## Emit deterministic live-CPU recipe shards
	@$(LOG_TARGET)
	@$(RECIPE_CONFORMANCE_PYTHON) tools/agent/scripts/recipe_conformance.py \
		plan --shards "$(RECIPE_CONFORMANCE_SHARDS)"

recipe-conformance-report: ## Assemble downloaded shard artifacts into one report
	@$(LOG_TARGET)
	@$(RECIPE_CONFORMANCE_PYTHON) tools/agent/scripts/recipe_conformance.py \
		--output-dir "$(RECIPE_CONFORMANCE_REPORT_DIR)" \
		report

recipe-conformance-eval: ## Evaluate one active recipe router (set RECIPE_CONFORMANCE_RECIPE)
	@$(LOG_TARGET)
	@if [ -z "$(RECIPE_CONFORMANCE_RECIPE)" ]; then \
		echo "RECIPE_CONFORMANCE_RECIPE is required"; \
		exit 2; \
	fi
	@$(RECIPE_CONFORMANCE_PYTHON) tools/agent/scripts/recipe_conformance.py \
		--output-dir "$(RECIPE_CONFORMANCE_REPORT_DIR)" \
		eval \
		--recipe "$(RECIPE_CONFORMANCE_RECIPE)" \
		--router-url "$(RECIPE_CONFORMANCE_ROUTER_URL)" \
		$(foreach binding,$(RECIPE_CONFORMANCE_ENTRYPOINTS),--entrypoint "$(binding)")

recipe-conformance-live-cpu: ## Build once and run live CPU probes (set RECIPE_CONFORMANCE_RECIPES)
	@$(LOG_TARGET)
	@if [ -z "$(RECIPE_CONFORMANCE_RECIPES)" ]; then \
		echo "RECIPE_CONFORMANCE_RECIPES is required"; \
		exit 2; \
	fi
	@$(MAKE) vllm-sr-router-build
	@RECIPES="$(RECIPE_CONFORMANCE_RECIPES)" \
		ROUTER_URL="$(RECIPE_CONFORMANCE_ROUTER_URL)" \
		REPORT_ROOT="$(RECIPE_CONFORMANCE_REPORT_DIR)" \
		bash e2e/testing/run_recipe_conformance.sh

recipe-conformance-live-cpu-all: ## Build once and run all maintained recipes
	@$(MAKE) recipe-conformance-live-cpu \
		RECIPE_CONFORMANCE_RECIPES="$$($(RECIPE_CONFORMANCE_PYTHON) tools/agent/scripts/recipe_conformance.py list --format csv)"

.PHONY: recipe-conformance-static recipe-conformance-plan \
	recipe-conformance-report \
	recipe-conformance-eval recipe-conformance-live-cpu \
	recipe-conformance-live-cpu-all
