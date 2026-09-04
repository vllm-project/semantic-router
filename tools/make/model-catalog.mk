# ====================== model-catalog.mk ======================
# = Unified provider/model catalog generation and validation  =
# =============================================================

.PHONY: model-catalog-generate model-catalog-check model-catalog-test

model-catalog-generate: ## Regenerate Router, CLI, Dashboard, and website catalog projections
	@.venv-agent/bin/python tools/catalog/generate_model_catalog.py

model-catalog-test: ## Run catalog compiler contract tests
	@.venv-agent/bin/python -m unittest discover -s tools/catalog/tests -p "test_*.py"

model-catalog-check: model-catalog-test ## Reject invalid catalog sources or stale generated projections
	@.venv-agent/bin/python tools/catalog/generate_model_catalog.py --check
