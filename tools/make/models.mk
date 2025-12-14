# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

##@ Models

# CI_MINIMAL_MODELS=true will download only the minimal set of models required for tests.
# Default behavior downloads the full set used for local development.
# Model configurations are in config/model_manager directory
#  full set: config/model_manager/models.yaml
#  minimal set: config/model_manager/models.minimal.yaml
#  LoRA and advanced embedding models: config/model_manager/models.lora.yaml

download-models:
download-models: ## Download models (full or minimal set depending on CI_MINIMAL_MODELS)
	@$(LOG_TARGET)
	@mkdir -p models
	# Delegate to Python Model Manager which handles CI_MINIMAL_MODELS env var
	PYTHONPATH=src python -m model_manager

# Explicit minimal download (useful for debugging CI mode locally)
download-models-minimal:
download-models-minimal: ## Pre-download minimal set of models for CI tests
	@mkdir -p models
	PYTHONPATH=src python -m model_manager --config config/model_manager/models.minimal.yaml

# Explicit full download
download-models-full:
download-models-full: ## Download all models used in local development and docs
	@mkdir -p models
	PYTHONPATH=src python -m model_manager --config config/model_manager/models.yaml

# Download only LoRA and advanced embedding models (for CI after minimal tests)
download-models-lora:
download-models-lora: ## Download LoRA adapters and advanced embedding models only
	@mkdir -p models
	PYTHONPATH=src python -m model_manager --config config/model_manager/models.lora.yaml

# Clean up minimal models to save disk space (for CI)
clean-minimal-models: ## Remove minimal models to save disk space
	@echo "Cleaning up minimal models to save disk space..."
	PYTHONPATH=src python -m model_manager --config config/model_manager/models.minimal.yaml --clean

# List configured models
list-models: ## List models configured in the default or specified environment
	@PYTHONPATH=src python -m model_manager --list
