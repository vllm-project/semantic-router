# ======== openvino.mk ========
# = Everything For OpenVINO  =
# ======== openvino.mk ========

OPENVINO_PYTHON ?= python3
OPENVINO_TEST_DEVICE ?= $(if $(MODEL_TEST_DEVICE),$(MODEL_TEST_DEVICE),CPU)
# The shared executor supplies a generic destination. Do not inherit the
# Candle-oriented file default from models.mk during standalone OpenVINO runs.
ifneq ($(filter environment command line override,$(origin MODEL_TEST_REPORT_DIR)),)
OPENVINO_TEST_REPORT_DIR ?= $(MODEL_TEST_REPORT_DIR)
else
OPENVINO_TEST_REPORT_DIR ?= $(CURDIR)/.agent-harness/model-tests/openvino-cpu
endif
OPENVINO_TEST_SOURCE_DIR ?= $(CURDIR)/models/openvino
OPENVINO_TEST_MODEL_DIR ?= $(CURDIR)/.agent-harness/openvino-models
OPENVINO_OWNED_FIXTURE_DIR ?= $(OPENVINO_TEST_REPORT_DIR)/owned-fixtures

define OPENVINO_TEST_ENV
OV_LIB_DIR=$$($(OPENVINO_PYTHON) -c "import openvino; print(openvino.__path__[0])")/libs; \
OV_TOK_DIR=$$($(OPENVINO_PYTHON) -c "import openvino_tokenizers; print(openvino_tokenizers.__path__[0])")/lib; \
export $(NATIVE_ENV); \
export LD_LIBRARY_PATH="$(CURDIR)/openvino-binding/build:$${OV_LIB_DIR}:$${OV_TOK_DIR}:$$LD_LIBRARY_PATH"; \
export OPENVINO_TOKENIZERS_LIB="$${OV_TOK_DIR}/libopenvino_tokenizers.so"; \
export OPENVINO_TEST_DEVICE="$(OPENVINO_TEST_DEVICE)"; \
export OPENVINO_TEST_REPORT_DIR="$(OPENVINO_TEST_REPORT_DIR)"; \
export OPENVINO_TEST_MANIFEST="$(OPENVINO_TEST_REPORT_DIR)/models.json"; \
export OPENVINO_OWNED_FIXTURE_DIR="$(OPENVINO_OWNED_FIXTURE_DIR)"; \
export SEMANTIC_ROUTER_OPENVINO_TEST_ARTIFACTS="$(OPENVINO_OWNED_FIXTURE_DIR)"; \
export OPENVINO_EMBEDDING_TOTAL_THREADS=2 OPENVINO_EMBEDDING_NUM_STREAMS=1 OPENVINO_EMBEDDING_NUM_REQUESTS=2; \
export OPENVINO_CLASSIFIER_TOTAL_THREADS=2 OPENVINO_CLASSIFIER_NUM_STREAMS=1 OPENVINO_CLASSIFIER_NUM_REQUESTS=2
endef

##@ OpenVINO

# Build OpenVINO binding C++ library
build-openvino-binding: ## Build OpenVINO C++ binding library
	@$(LOG_TARGET)
	@echo "Building OpenVINO C++ binding library..."
	@mkdir -p openvino-binding/build
	@cd openvino-binding/build && \
		cmake .. -DPython3_EXECUTABLE="$$(command -v $(OPENVINO_PYTHON))" && \
		$(MAKE) -j$$(nproc) COLOR= VERBOSE=
	@echo "✅ OpenVINO binding built: openvino-binding/build/libopenvino_semantic_router.so"

test-openvino-binding: verify-openvino-binding ## Run the required published Vela inference contract

# Clean OpenVINO build artifacts
clean-openvino-binding: ## Clean OpenVINO build artifacts
	@echo "Cleaning OpenVINO build artifacts..."
	@rm -rf openvino-binding/build
	@echo "✅ OpenVINO build artifacts cleaned"

# Run specific OpenVINO test
# Example: make test-openvino-specific TEST_NAME=TestEmbeddings
test-openvino-specific: build-openvino-binding ## Run a legacy optional test against manually prepared artifacts (TEST_NAME=TestName)
	@$(LOG_TARGET)
	@if [ -z "$(TEST_NAME)" ]; then \
		echo "ERROR: TEST_NAME not specified"; \
		echo "Usage: make test-openvino-specific TEST_NAME=TestEmbeddings"; \
		exit 1; \
	fi
	@echo "Running OpenVINO test: $(TEST_NAME)"
	@OV_LIB_DIR=$$(python3 -c "import openvino; print(openvino.__path__[0])" 2>/dev/null)/libs; \
		OV_TOK_DIR=$$(python3 -c "import openvino_tokenizers; print(openvino_tokenizers.__path__[0])" 2>/dev/null)/lib; \
		export $(NATIVE_ENV) && \
		export LD_LIBRARY_PATH="$(CURDIR)/openvino-binding/build:$${OV_LIB_DIR}:$${OV_TOK_DIR}:$$LD_LIBRARY_PATH" && \
		cd openvino-binding && CGO_ENABLED=1 go test -v -timeout 10m -run "^$(TEST_NAME)$$"

# Keep the upstream three same-process race repetitions: a fresh process per
# repetition would miss ownership state retained between model generations.
verify-openvino-owned-binding: build-openvino-binding
	@mkdir -p "$(OPENVINO_TEST_REPORT_DIR)/owned-bindings" "$(OPENVINO_TEST_REPORT_DIR)/owned-runtime"
	@rm -f "$(OPENVINO_TEST_REPORT_DIR)/inference.json" "$(OPENVINO_TEST_REPORT_DIR)/tests.jsonl" \
		"$(OPENVINO_TEST_REPORT_DIR)/owned-bindings/"*.jsonl "$(OPENVINO_TEST_REPORT_DIR)/owned-runtime/"*.jsonl
	@$(OPENVINO_PYTHON) openvino-binding/scripts/create_owned_fixture.py "$(OPENVINO_OWNED_FIXTURE_DIR)"
	@$(OPENVINO_TEST_ENV); \
	cd openvino-binding && \
	CGO_ENABLED=1 go test -json -list '^TestOwned' . > "$(OPENVINO_TEST_REPORT_DIR)/owned-bindings/discovery.jsonl" 2>&1 && \
	CGO_ENABLED=1 go test -json -race -count=3 -run '^TestOwned' -timeout 5m . \
		> "$(OPENVINO_TEST_REPORT_DIR)/owned-bindings/tests.jsonl" 2>&1; status=$$?; \
	cat "$(OPENVINO_TEST_REPORT_DIR)/owned-bindings/"*.jsonl; exit $$status

verify-openvino-runtime: verify-openvino-owned-binding rust-ci build-onnx-binding
	@$(OPENVINO_TEST_ENV); \
	cd src/semantic-router && \
	CGO_ENABLED=1 go test -json -tags openvino -list '^(TestOwnedOpenVINORuntimeIntegration|TestOpenVINOQualifiedFixtureMetadata)$$' ./pkg/modelruntime/native \
		> "$(OPENVINO_TEST_REPORT_DIR)/owned-runtime/discovery.jsonl" 2>&1 && \
	CGO_ENABLED=1 go test -json -race -count=3 -tags openvino -run '^(TestOwnedOpenVINORuntimeIntegration|TestOpenVINOQualifiedFixtureMetadata)$$' -timeout 5m ./pkg/modelruntime/native \
		> "$(OPENVINO_TEST_REPORT_DIR)/owned-runtime/tests.jsonl" 2>&1; status=$$?; \
	cat "$(OPENVINO_TEST_REPORT_DIR)/owned-runtime/"*.jsonl; exit $$status

# The standalone binding image exercises C++ ownership and real weights without
# claiming the router integration; native CI also requires verify-openvino-runtime.
verify-openvino-published: verify-openvino-owned-binding convert-openvino-test-models
	@rm -f "$(OPENVINO_TEST_REPORT_DIR)/inference.json"
	@$(OPENVINO_TEST_ENV); \
	export OPENVINO_EMBEDDING_INFER_POOL_SIZE=1 OPENVINO_EMBEDDING_NUM_REQUESTS=1; \
	export OPENVINO_CLASSIFIER_INFER_POOL_SIZE=1 OPENVINO_CLASSIFIER_NUM_REQUESTS=1; \
	cd openvino-binding && CGO_ENABLED=1 go test -json -count=1 -timeout 30m \
		-tags published_model_tests -run '^TestPublishedOpenVINO$$' . \
		> "$(OPENVINO_TEST_REPORT_DIR)/tests.jsonl" 2>&1; status=$$?; \
	cat "$(OPENVINO_TEST_REPORT_DIR)/tests.jsonl"; \
	[ $$status -eq 0 ] && test -s "$(OPENVINO_TEST_REPORT_DIR)/inference.json"

verify-openvino-binding: verify-openvino-runtime verify-openvino-published ## Require owned runtime contracts and published Vela inference on the selected OpenVINO device

# Benchmark OpenVINO vs Candle binding
benchmark-openvino-vs-candle: build-openvino-binding rust ## Benchmark manually prepared OpenVINO vs Candle artifacts
	@$(LOG_TARGET)
	@echo "Running OpenVINO vs Candle benchmark..."
	@export $(NATIVE_ENV) && \
		export LD_LIBRARY_PATH="$(CURDIR)/openvino-binding/build:$$LD_LIBRARY_PATH" && \
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
