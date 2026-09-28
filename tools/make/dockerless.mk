# ======== local.mk =============
# = Everything For Dockerless   =
# ======== local.mk ============

##@ Dockerless

local-up-router: ## Start local semantic-router
	echo "Starting local semantic-router with Envoy proxy..."
	@tools/dev/local-up-router.sh
