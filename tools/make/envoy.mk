# ======== envoy.mk ========
# = Everything For envoy   =
# ======== envoy.mk ========

##@ Envoy

prepare-envoy: $(FUNC_E) ## Install func-e for managing Envoy versions
	@$(LOG_TARGET)
	@$(FUNC_E) --version

run-envoy: $(FUNC_E) ## Run Envoy proxy with the configured settings
	@$(LOG_TARGET)
	@echo "Starting Envoy..."
	$(FUNC_E) run --config-path config/envoy.yaml --component-log-level "ext_proc:trace,router:trace,http:trace"
