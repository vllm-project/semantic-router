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
