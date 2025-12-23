# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

##@ Models

# Models are automatically downloaded by the router at startup in production.
# For testing, we use the router's --download-only flag to download models and exit.

# Download models by running the router with --download-only flag
download-models: ## Download models using router's built-in download logic
	@echo "📦 Downloading models via router..."
	@echo ""
	@$(MAKE) build-router
	@echo ""
	@echo "Running router with --download-only flag..."
	@echo "This may take a few minutes depending on your network speed..."
	@./bin/router -config=config/config.yaml --download-only
	@echo ""
	@echo "✅ Models downloaded successfully"

download-models-lora: ## Download LoRA models (same as download-models now)
	@$(MAKE) download-models

clean-minimal-models: ## No-op target for backward compatibility
	@echo "ℹ️  This target is no longer needed"
