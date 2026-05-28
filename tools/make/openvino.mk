# ======== openvino.mk ========
# = Everything For OpenVINO  =
# ======== openvino.mk ========

##@ OpenVINO

# Build OpenVINO binding C++ library
build-openvino-binding: ## Build OpenVINO C++ binding library
	@$(LOG_TARGET)
	@echo "Building OpenVINO C++ binding library..."
	@mkdir -p openvino-binding/build
	@cd openvino-binding/build && \
		cmake .. && \
		$(MAKE) -j$$(nproc) COLOR= VERBOSE=
	@echo "✅ OpenVINO binding built: openvino-binding/build/libopenvino_semantic_router.so"

# Test OpenVINO binding - depends on models being converted
test-openvino-binding: build-openvino-binding convert-openvino-test-models ## Run Go tests for OpenVINO binding
	@$(LOG_TARGET)
	@echo "Running OpenVINO binding Go unit tests..."
	@echo "================================================================"
	@OV_LIB_DIR=$$(python3 -c "import openvino; print(openvino.__path__[0])" 2>/dev/null)/libs; \
		OV_TOK_DIR=$$(python3 -c "import openvino_tokenizers; print(openvino_tokenizers.__path__[0])" 2>/dev/null)/lib; \
		export LD_LIBRARY_PATH=$${PWD}/openvino-binding/build:$${OV_LIB_DIR}:$${OV_TOK_DIR}:$$LD_LIBRARY_PATH && \
		cd openvino-binding && CGO_ENABLED=1 go test -v -timeout 10m
	@echo "================================================================"
	@echo "✅ OpenVINO binding tests passed"

# Clean OpenVINO build artifacts
clean-openvino-binding: ## Clean OpenVINO build artifacts
	@echo "Cleaning OpenVINO build artifacts..."
	@rm -rf openvino-binding/build
	@echo "✅ OpenVINO build artifacts cleaned"

# Run specific OpenVINO test
# Example: make test-openvino-specific TEST_NAME=TestEmbeddings
test-openvino-specific: build-openvino-binding convert-openvino-test-models ## Run specific OpenVINO test (TEST_NAME=TestName)
	@$(LOG_TARGET)
	@if [ -z "$(TEST_NAME)" ]; then \
		echo "ERROR: TEST_NAME not specified"; \
		echo "Usage: make test-openvino-specific TEST_NAME=TestEmbeddings"; \
		exit 1; \
	fi
	@echo "Running OpenVINO test: $(TEST_NAME)"
	@OV_LIB_DIR=$$(python3 -c "import openvino; print(openvino.__path__[0])" 2>/dev/null)/libs; \
		OV_TOK_DIR=$$(python3 -c "import openvino_tokenizers; print(openvino_tokenizers.__path__[0])" 2>/dev/null)/lib; \
		export LD_LIBRARY_PATH=$${PWD}/openvino-binding/build:$${OV_LIB_DIR}:$${OV_TOK_DIR}:$$LD_LIBRARY_PATH && \
		cd openvino-binding && CGO_ENABLED=1 go test -v -timeout 10m -run "^$(TEST_NAME)$$"

# Verify OpenVINO binding with real model inference
verify-openvino-binding: build-openvino-binding convert-openvino-test-models ## Verify OpenVINO binding uses real model inference
	@$(LOG_TARGET)
	@echo "Verifying OpenVINO binding with real model inference..."
	@echo "================================================================"
	@OV_LIB_DIR=$$(python3 -c "import openvino; print(openvino.__path__[0])" 2>/dev/null)/libs; \
		OV_TOK_DIR=$$(python3 -c "import openvino_tokenizers; print(openvino_tokenizers.__path__[0])" 2>/dev/null)/lib; \
		export LD_LIBRARY_PATH=$${PWD}/openvino-binding/build:$${OV_LIB_DIR}:$${OV_TOK_DIR}:$$LD_LIBRARY_PATH && \
		cd openvino-binding && go run verify_tests_are_real.go
	@echo "================================================================"
	@echo "✅ OpenVINO binding verification passed"

# Benchmark OpenVINO vs Candle binding
benchmark-openvino-vs-candle: build-openvino-binding rust convert-openvino-test-models ## Benchmark OpenVINO vs Candle
	@$(LOG_TARGET)
	@echo "Running OpenVINO vs Candle benchmark..."
	@export LD_LIBRARY_PATH=$${PWD}/openvino-binding/build:$${PWD}/candle-binding/target/release:$$LD_LIBRARY_PATH && \
		cd openvino-binding/bench && go run mmbert_classifier_bench.go


# Run classifier benchmark script (supports ARGS="--run-only", etc.)
benchmark-openvino-classifier: build-openvino-binding rust ## Run OpenVINO classifier benchmark script
	@$(LOG_TARGET)
	@echo "Running OpenVINO classifier benchmark script..."
	@bash openvino-binding/scripts/build_and_run_mmbert_classifier_bench.sh $(ARGS)

# Run embedding benchmark script (supports ARGS="--run-only --length-profile fixed-128", etc.)
benchmark-openvino-embedding: build-openvino-binding rust ## Run OpenVINO embedding benchmark script
	@$(LOG_TARGET)
	@echo "Running OpenVINO embedding benchmark script..."
	@bash openvino-binding/scripts/build_and_run_mmbert_embedding_bench.sh $(ARGS)
