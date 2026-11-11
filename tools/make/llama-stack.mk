# ======== llama-stack.mk ========
# = Everything For Llama Stack  =
# ======== llama-stack.mk ========

##@ Llama Stack

LLAMA_STACK_CONTAINER_NAME ?= llama-stack-vectorstore
LLAMA_STACK_PORT ?= 8321
LLAMA_STACK_IMAGE ?= ogxai/distribution-starter:1.5.0
LLAMA_STACK_DATA_DIR ?= /tmp/llama-stack-data
LLAMA_STACK_EMBEDDING_MODEL_ID ?= all-MiniLM-L6-v2
LLAMA_STACK_EMBEDDING_DIMENSION ?= 384
LLAMA_STACK_SEARCH_TYPE ?= hybrid
LLAMA_STACK_VECTOR_IO_PROVIDER ?= sqlite-vec

# Llama Stack container management. The image is the starter distribution of
# OGX, the project Llama Stack became; the container registers the embedding
# model and the default vector_io provider in a copy of its config.
start-llama-stack: ## Start Llama Stack container with a hybrid-search vector_io provider for testing
	@$(LOG_TARGET)
	@if $(CONTAINER_RUNTIME) ps --filter "name=$(LLAMA_STACK_CONTAINER_NAME)" --format "{{.Names}}" | grep -q $(LLAMA_STACK_CONTAINER_NAME); then \
		echo "Llama Stack container is already running"; \
	else \
		$(CONTAINER_RUNTIME) rm $(LLAMA_STACK_CONTAINER_NAME) 2>/dev/null || true; \
		mkdir -p $(LLAMA_STACK_DATA_DIR); \
		echo "Starting Llama Stack with $(LLAMA_STACK_VECTOR_IO_PROVIDER) vector_io provider..."; \
		$(CONTAINER_RUNTIME) run -d \
			--name $(LLAMA_STACK_CONTAINER_NAME) \
			--user $$(id -u):0 \
			-p $(LLAMA_STACK_PORT):$(LLAMA_STACK_PORT) \
			-v $(LLAMA_STACK_DATA_DIR):/.ogx \
			-e RUN_CONFIG_PATH=/tmp/config.yaml \
			--entrypoint sh \
			$(LLAMA_STACK_IMAGE) \
			-c "python3 -c \"import pathlib, yaml, ogx; \
			src = pathlib.Path(ogx.__file__).parent / 'distributions/starter/config.yaml'; \
			cfg = yaml.safe_load(src.read_text()); \
			cfg['registered_resources']['models'].append({'model_id': '$(LLAMA_STACK_EMBEDDING_MODEL_ID)', 'provider_id': 'sentence-transformers', 'provider_model_id': 'sentence-transformers/$(LLAMA_STACK_EMBEDDING_MODEL_ID)', 'model_type': 'embedding', 'metadata': {'embedding_dimension': $(LLAMA_STACK_EMBEDDING_DIMENSION)}}); \
			cfg['vector_stores']['default_provider_id'] = '$(LLAMA_STACK_VECTOR_IO_PROVIDER)'; \
			pathlib.Path('/tmp/config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))\" && \
			exec /usr/local/bin/ogx-entrypoint.sh --port $(LLAMA_STACK_PORT)"; \
		echo "Waiting for Llama Stack to be ready..."; \
		for i in $$(seq 1 60); do \
			if curl -sf http://localhost:$(LLAMA_STACK_PORT)/v1/health > /dev/null 2>&1; then \
				echo "Llama Stack is ready at localhost:$(LLAMA_STACK_PORT)"; \
				echo "Verifying $(LLAMA_STACK_VECTOR_IO_PROVIDER) provider..."; \
				curl -sf http://localhost:$(LLAMA_STACK_PORT)/v1/providers 2>/dev/null | \
					python3 -c "import sys,json; [print(f'  vector_io: {p[\"provider_id\"]} ({p[\"provider_type\"]})') for p in json.load(sys.stdin).get('data',[]) if p.get('api')=='vector_io']" 2>/dev/null || true; \
				break; \
			fi; \
			if [ $$i -eq 60 ]; then \
				echo "Warning: Llama Stack health check timed out (may still be starting)"; \
				$(CONTAINER_RUNTIME) logs $(LLAMA_STACK_CONTAINER_NAME) 2>&1 | tail -10; \
			fi; \
			sleep 2; \
		done; \
	fi

stop-llama-stack: ## Stop and remove Llama Stack container
	@$(LOG_TARGET)
	@$(CONTAINER_RUNTIME) stop $(LLAMA_STACK_CONTAINER_NAME) || true
	@$(CONTAINER_RUNTIME) rm $(LLAMA_STACK_CONTAINER_NAME) || true
	@echo "Llama Stack container stopped and removed"

restart-llama-stack: stop-llama-stack start-llama-stack ## Restart Llama Stack container

llama-stack-status: ## Show status of Llama Stack container
	@$(LOG_TARGET)
	@if $(CONTAINER_RUNTIME) ps --filter "name=$(LLAMA_STACK_CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q $(LLAMA_STACK_CONTAINER_NAME); then \
		echo "Llama Stack container is running:"; \
		$(CONTAINER_RUNTIME) ps --filter "name=$(LLAMA_STACK_CONTAINER_NAME)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
		echo ""; \
		echo "Health check:"; \
		curl -sf http://localhost:$(LLAMA_STACK_PORT)/v1/health && echo " OK" || echo " UNHEALTHY"; \
	else \
		echo "Llama Stack container is not running"; \
		echo "Run 'make start-llama-stack' to start it"; \
	fi

clean-llama-stack: stop-llama-stack ## Clean up Llama Stack data
	@$(LOG_TARGET)
	@echo "Cleaning up Llama Stack data..."
	@sudo rm -rf $(LLAMA_STACK_DATA_DIR) || rm -rf $(LLAMA_STACK_DATA_DIR)
	@echo "Llama Stack data directory cleaned"

# Test vector store with Llama Stack backend
test-llama-stack-vectorstore: start-llama-stack ## Run Llama Stack vector store integration tests
	@$(LOG_TARGET)
	@echo "Testing vector store with Llama Stack backend..."
	@export SKIP_LLAMA_STACK_TESTS=false && \
	export LLAMA_STACK_ENDPOINT=http://localhost:$(LLAMA_STACK_PORT) && \
	export LLAMA_STACK_EMBEDDING_MODEL=$${LLAMA_STACK_EMBEDDING_MODEL:-sentence-transformers/$(LLAMA_STACK_EMBEDDING_MODEL_ID)} && \
	export LLAMA_STACK_SEARCH_TYPE=$(LLAMA_STACK_SEARCH_TYPE) && \
	export SR_TEST_MODE=true && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v -count=1 \
		./pkg/vectorstore/ -ginkgo.focus="LlamaStack"
	@echo "Consider running 'make stop-llama-stack' when done testing"
