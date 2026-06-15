# ======== qdrant.mk ========
# = Everything For Qdrant   =
# ======== qdrant.mk ========

##@ Qdrant

QDRANT_IMAGE ?= qdrant/qdrant:latest
QDRANT_CONTAINER ?= qdrant-semantic-router

start-qdrant: ## Start Qdrant container for testing
	@$(LOG_TARGET)
	@mkdir -p /tmp/qdrant-data
	@$(CONTAINER_RUNTIME) run -d \
		--name $(QDRANT_CONTAINER) \
		-p 6333:6333 \
		-p 6334:6334 \
		-v /tmp/qdrant-data:/qdrant/storage:z \
		$(QDRANT_IMAGE)
	@echo "Waiting for Qdrant to be ready (up to 30s)..."
	@elapsed=0; \
	while [ $$elapsed -lt 30 ]; do \
		if curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then \
			echo "Qdrant healthy after $${elapsed}s"; \
			break; \
		fi; \
		if ! $(CONTAINER_RUNTIME) ps --filter "name=$(QDRANT_CONTAINER)" --format '{{.Names}}' | grep -q $(QDRANT_CONTAINER); then \
			echo "ERROR: Qdrant container exited unexpectedly"; \
			$(CONTAINER_RUNTIME) logs $(QDRANT_CONTAINER) 2>&1 | tail -20 || true; \
			exit 1; \
		fi; \
		sleep 2; \
		elapsed=$$((elapsed + 2)); \
		echo "  ... still waiting ($${elapsed}s elapsed)"; \
	done; \
	if [ $$elapsed -ge 30 ]; then \
		echo "ERROR: Qdrant did not become healthy within 30s"; \
		$(CONTAINER_RUNTIME) logs $(QDRANT_CONTAINER) 2>&1 | tail -30 || true; \
		exit 1; \
	fi
	@echo "Qdrant available at localhost:6334 (gRPC) / localhost:6333 (REST)"

stop-qdrant: ## Stop and remove Qdrant container
	@$(LOG_TARGET)
	@$(CONTAINER_RUNTIME) stop $(QDRANT_CONTAINER) || true
	@$(CONTAINER_RUNTIME) rm $(QDRANT_CONTAINER) || true
	@rm -rf /tmp/qdrant-data 2>/dev/null || sudo rm -rf /tmp/qdrant-data 2>/dev/null || true
	@echo "Qdrant container stopped and removed"

restart-qdrant: stop-qdrant start-qdrant ## Restart Qdrant container

qdrant-status: ## Show status of Qdrant container
	@$(LOG_TARGET)
	@if $(CONTAINER_RUNTIME) ps --filter "name=$(QDRANT_CONTAINER)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q $(QDRANT_CONTAINER); then \
		echo "Qdrant container is running:"; \
		$(CONTAINER_RUNTIME) ps --filter "name=$(QDRANT_CONTAINER)" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	else \
		echo "Qdrant container is not running"; \
		echo "Run 'make start-qdrant' to start it"; \
	fi

clean-qdrant: stop-qdrant ## Clean up Qdrant data
	@$(LOG_TARGET)
	@echo "Cleaning up Qdrant data..."
	@rm -rf /tmp/qdrant-data 2>/dev/null || sudo rm -rf /tmp/qdrant-data 2>/dev/null || true
	@echo "Qdrant data directory cleaned"

test-qdrant: start-qdrant rust ## Run Qdrant integration tests
	@$(LOG_TARGET)
	@echo "Running Qdrant integration tests..."
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release:$${PWD}/ml-binding/target/release:$${PWD}/nlp-binding/target/release && \
	export SKIP_QDRANT_TESTS=false && \
		cd src/semantic-router && CGO_ENABLED=1 go test -v \
		./pkg/cache/ \
		./pkg/memory/ \
		./pkg/vectorstore/ \
		./pkg/routerreplay/store/ \
		-run Qdrant
	@echo "Consider running 'make stop-qdrant' when done testing"
