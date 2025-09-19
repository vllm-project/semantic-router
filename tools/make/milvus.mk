# ======== milvus.mk ========
# = Everything For milvus   =
# ======== milvus.mk ========

# Milvus container management
start-milvus:
	@$(LOG_TARGET)
	@echo "Starting Milvus container for testing with $(CONTAINER_RUNTIME)..."
	@mkdir -p /tmp/milvus-data
	@$(CONTAINER_RUNTIME) run -d \
		--name milvus-semantic-cache \
		--security-opt seccomp:unconfined \
		-e ETCD_USE_EMBED=true \
		-e ETCD_DATA_DIR=/var/lib/milvus/etcd \
		-e ETCD_CONFIG_PATH=/milvus/configs/advanced/etcd.yaml \
		-e COMMON_STORAGETYPE=local \
		-e CLUSTER_ENABLED=false \
		-p 19530:19530 \
		-p 9091:9091 \
		-v /tmp/milvus-data:/var/lib/milvus \
		milvusdb/milvus:v2.3.3 \
		milvus run standalone
	@echo "Waiting for Milvus to be ready..."
	@sleep 15
	@echo "Milvus should be available at localhost:19530"

stop-milvus:
	@$(LOG_TARGET)
	@echo "Stopping Milvus container..."
	@$(CONTAINER_RUNTIME) stop milvus-semantic-cache || true
	@$(CONTAINER_RUNTIME) rm milvus-semantic-cache || true
	@sudo rm -rf /tmp/milvus-data || true
	@echo "Milvus container stopped and removed"

restart-milvus: stop-milvus start-milvus

milvus-status:
	@$(LOG_TARGET)
	@echo "Checking Milvus container status..."
	@if $(CONTAINER_RUNTIME) ps --filter "name=milvus-semantic-cache" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q milvus-semantic-cache; then \
		echo "Milvus container is running:"; \
		$(CONTAINER_RUNTIME) ps --filter "name=milvus-semantic-cache" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	else \
		echo "Milvus container is not running"; \
		echo "Run 'make start-milvus' to start it"; \
	fi

clean-milvus: stop-milvus
	@$(LOG_TARGET)
	@echo "Cleaning up Milvus data..."
	@sudo rm -rf milvus-data || rm -rf milvus-data
	@echo "Milvus data directory cleaned"

# Test semantic cache with Milvus backend
test-milvus-cache: start-milvus rust
	@$(LOG_TARGET)
	@echo "Testing semantic cache with Milvus backend..."
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
		cd src/semantic-router && CGO_ENABLED=1 go test -tags=milvus -v ./pkg/cache/
	@echo "Consider running 'make stop-milvus' when done testing"

# Test semantic-router with Milvus enabled
test-semantic-router-milvus: build-router start-milvus
	@$(LOG_TARGET)
	@echo "Testing semantic-router with Milvus cache backend..."
	@export LD_LIBRARY_PATH=$${PWD}/candle-binding/target/release && \
		cd src/semantic-router && CGO_ENABLED=1 go test -tags=milvus -v ./...
	@echo "Consider running 'make stop-milvus' when done testing"

# Milvus UI (Attu) management
start-milvus-ui: ## Start Attu UI to browse Milvus data
	@$(LOG_TARGET)
	@echo "Starting Attu (Milvus UI) with $(CONTAINER_RUNTIME)..."
	@$(CONTAINER_RUNTIME) run -d \
		--name milvus-ui \
		--add-host=host.docker.internal:host-gateway \
		-e MILVUS_URL=host.docker.internal:19530 \
		-p 18000:3000 \
		zilliz/attu:v2.3.5
	@echo "Waiting for Attu to be ready..."
	@sleep 3
	@echo "Open UI: http://localhost:18000 (Milvus at host.docker.internal:19530)"

stop-milvus-ui:
	@$(LOG_TARGET)
	@echo "Stopping Attu (Milvus UI) container..."
	@$(CONTAINER_RUNTIME) stop milvus-ui || true
	@$(CONTAINER_RUNTIME) rm milvus-ui || true
	@echo "Attu container stopped and removed"