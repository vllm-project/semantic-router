# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

##@ Models

# Models are now automatically downloaded by the router at startup.
# The router uses the mom_registry in config/config.yaml to determine
# which models to download from HuggingFace.
#
# No manual model download is required for development or CI.
# The router will check for missing models and download them automatically
# using the Go-based modeldownload package.

# Empty target for backward compatibility
# Models are now downloaded automatically by the router at startup
download-models: ## (Deprecated) Models are now downloaded automatically at startup
	@echo "ℹ️  Models are now downloaded automatically by the router at startup"
	@echo "ℹ️  No manual download required - the router will handle it"

download-models-lora: ## (Deprecated) Models are now downloaded automatically at startup
	@echo "ℹ️  Models are now downloaded automatically by the router at startup"
	@echo "ℹ️  No manual download required - the router will handle it"

clean-minimal-models: ## (Deprecated) No-op target for backward compatibility
	@echo "ℹ️  This target is no longer needed"
