# ======== envoy.mk ========
# = Everything For envoy   =
# ======== envoy.mk ========

##@ Envoy

prepare-envoy: ## Install func-e for managing Envoy versions
	@$(LOG_TARGET)
	curl https://func-e.io/install.sh | sudo bash -s -- -b /usr/local/bin

run-envoy: ## Run Envoy proxy with the configured settings
	@$(LOG_TARGET)
	@echo "Checking for func-e..."
	@if ! command -v func-e >/dev/null 2>&1; then \
		echo "func-e not found, installing..."; \
		$(MAKE) prepare-envoy; \
	fi
	@echo "Starting Envoy..."
	func-e run --config-path config/envoy.yaml --component-log-level "ext_proc:trace,router:trace,http:trace"
