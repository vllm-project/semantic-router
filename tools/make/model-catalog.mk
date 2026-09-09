# ====================== model-catalog.mk ======================
# = Unified provider/model catalog generation and validation  =
# =============================================================

MODEL_CATALOG_PYTHON ?= $(if $(wildcard $(CURDIR)/.venv-agent/bin/python),$(CURDIR)/.venv-agent/bin/python,python3)

.PHONY: model-catalog-generate model-catalog-check model-catalog-test \
	model-catalog-audit model-catalog-boundary-check \
	model-catalog-package-stage model-catalog-package-check

model-catalog-generate: ## Regenerate the built-in snapshot and Router/public projections
	@$(MODEL_CATALOG_PYTHON) tools/catalog/generate_model_catalog.py

model-catalog-package-stage: model-catalog-generate ## Stage ignored CLI package assets from the built-in snapshot
	@$(MODEL_CATALOG_PYTHON) tools/release/stage_model_catalog_package.py

model-catalog-package-check: ## Verify the staged CLI package assets byte for byte
	@$(MODEL_CATALOG_PYTHON) tools/release/stage_model_catalog_package.py --check

model-catalog-boundary-check: ## Reject checked-in consumer mirrors
	@test ! -e dashboard/frontend/src/generated/modelCatalog.json || { \
		echo "Dashboard must import website/static/model-catalog/catalog.json directly"; \
		exit 1; \
	}
	@tracked="$$(git ls-files src/vllm-sr/cli/model_assets | awk '$$0 != "src/vllm-sr/cli/model_assets/__init__.py"')"; \
		test -z "$$tracked" || { \
			echo "CLI model_assets version trees are build-only staging:"; \
			echo "$$tracked"; \
			exit 1; \
		}

model-catalog-test: ## Run catalog compiler contract tests
	@$(MODEL_CATALOG_PYTHON) -m unittest discover -s tools/catalog/tests -p "test_*.py"

model-catalog-check: model-catalog-test model-catalog-boundary-check ## Reject invalid/incomplete sources, mirrors, or stale projections
	@$(MODEL_CATALOG_PYTHON) tools/catalog/generate_model_catalog.py --check
	@$(MODEL_CATALOG_PYTHON) tools/catalog/audit_model_catalog.py --require-min-evaluations-per-model 5 >/dev/null

model-catalog-audit: ## Report authored catalog evaluation completeness (non-blocking by default)
	@$(MODEL_CATALOG_PYTHON) tools/catalog/audit_model_catalog.py $(MODEL_CATALOG_AUDIT_ARGS)
