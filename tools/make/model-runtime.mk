# ======== model-runtime.mk ========
# = Built-in model runtime (src/model-runtime) =
# ======== model-runtime.mk ========

MODEL_RUNTIME_DIR := src/model-runtime
MODEL_RUNTIME_OPENAPI := $(MODEL_RUNTIME_DIR)/vllm_sr_runtime/api/openapi.yaml
MODEL_RUNTIME_PYTHON ?= $(AGENT_PYTHON)
MODEL_RUNTIME_TORCH ?= torch==2.10.0
MODEL_RUNTIME_TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
MODEL_RUNTIME_CLIENT_DIR := src/semantic-router/pkg/modelservice/api
OAPI_CODEGEN_VERSION ?= v2.4.1

##@ Model runtime

model-runtime-install: ## Install the model runtime with CPU PyTorch and its test extras
	@$(LOG_TARGET)
	@"$(MODEL_RUNTIME_PYTHON)" -m pip install "$(MODEL_RUNTIME_TORCH)" --index-url "$(MODEL_RUNTIME_TORCH_INDEX)"
	@"$(MODEL_RUNTIME_PYTHON)" -m pip install -e "$(MODEL_RUNTIME_DIR)[test,reference]"

model-runtime-test: ## Run the model runtime tests on CPU (tiny fixtures; GPU cases skip)
	@$(LOG_TARGET)
	@cd $(MODEL_RUNTIME_DIR) && HF_HUB_OFFLINE=1 "$(MODEL_RUNTIME_PYTHON)" -m pytest -q -p no:cacheprovider tests

model-runtime-client-generate: ## Regenerate the router's Go client from the runtime OpenAPI contract
	@$(LOG_TARGET)
	@cd $(MODEL_RUNTIME_CLIENT_DIR) && go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen@$(OAPI_CODEGEN_VERSION) \
		-config oapi-codegen.yaml $(CURDIR)/$(MODEL_RUNTIME_OPENAPI)

model-runtime-client-check: ## Fail when the generated Go client differs from the OpenAPI contract
	@$(LOG_TARGET)
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
		cp $(MODEL_RUNTIME_CLIENT_DIR)/openapi.gen.go "$$tmp/openapi.gen.go" && \
		$(MAKE) --no-print-directory -f tools/make/common.mk -f tools/make/model-runtime.mk model-runtime-client-generate && \
		if ! diff -u "$$tmp/openapi.gen.go" $(MODEL_RUNTIME_CLIENT_DIR)/openapi.gen.go; then \
			cp "$$tmp/openapi.gen.go" $(MODEL_RUNTIME_CLIENT_DIR)/openapi.gen.go; \
			echo "the generated model runtime client is stale; run make model-runtime-client-generate"; exit 1; \
		fi

.PHONY: model-runtime-install model-runtime-test model-runtime-client-generate model-runtime-client-check
