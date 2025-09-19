# ======== envoy.mk ========
# = Everything For envoy   =
# ======== envoy.mk ========

# Prepare Envoy
prepare-envoy:
	@$(LOG_TARGET)
	curl https://func-e.io/install.sh | sudo bash -s -- -b /usr/local/bin

# Run Envoy proxy
run-envoy:
	@$(LOG_TARGET)
	@echo "Checking for func-e..."
	@if ! command -v func-e >/dev/null 2>&1; then \
		echo "func-e not found, installing..."; \
		$(MAKE) prepare-envoy; \
	fi
	@echo "Starting Envoy..."
	func-e run --config-path config/envoy.yaml --component-log-level "ext_proc:trace,router:trace,http:trace"
