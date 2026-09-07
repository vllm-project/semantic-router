# ====================== model-catalog.mk ======================
# = Unified provider/model catalog generation and validation  =
# =============================================================

MODEL_CATALOG_PYTHON ?= $(if $(wildcard $(CURDIR)/.venv-agent/bin/python),$(CURDIR)/.venv-agent/bin/python,python3)

.PHONY: model-catalog-generate model-catalog-check model-catalog-test model-catalog-audit

model-catalog-generate: ## Regenerate Router, CLI, Dashboard, and website catalog projections
	@$(MODEL_CATALOG_PYTHON) tools/catalog/generate_model_catalog.py

model-catalog-test: ## Run catalog compiler contract tests
	@$(MODEL_CATALOG_PYTHON) -m unittest discover -s tools/catalog/tests -p "test_*.py"

model-catalog-check: model-catalog-test ## Reject invalid/incomplete catalog sources or stale generated projections
	@$(MODEL_CATALOG_PYTHON) tools/catalog/generate_model_catalog.py --check
	@$(MODEL_CATALOG_PYTHON) tools/catalog/audit_model_catalog.py --require-min-evaluations-per-model 5 >/dev/null

model-catalog-audit: ## Report authored catalog evaluation completeness (non-blocking by default)
	@$(MODEL_CATALOG_PYTHON) tools/catalog/audit_model_catalog.py $(MODEL_CATALOG_AUDIT_ARGS)
