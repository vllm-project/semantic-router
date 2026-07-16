# ======== rust.mk ========
# = Everything For rust   =
# ======== rust.mk ========

# Default GPU device for testing (can be overridden: TEST_GPU_DEVICE=3 make test-rust)
TEST_GPU_DEVICE ?= 2

# Rust lib unit tests that are safe for PR CI:
# - no downloaded model assets
# - no GPU/CUDA requirement
# - deterministic pure Rust/FFI helper logic
#
# Keep this list explicit. Do not include tests whose fixtures initialize
# models from ../models unless those tests have been converted to skip cleanly.
RUST_CI_LIB_TESTS ?= \
	core::tokenization_test::test_tokenization_config_default \
	core::tokenization_test::test_tokenization_config_custom \
	ffi::embedding_test::test_truncate_embedding_renormalizes_prefix \
	model_architectures::embedding::multimodal_embedding::tests::test_siglip_vision_encoder_loads_with_head_weights \
	model_architectures::embedding::multimodal_embedding::tests::test_siglip_vision_encoder_requires_pooling_head

test-rust-ci:
	@$(LOG_TARGET)
	@echo "Running CI-safe Rust lib unit tests (CPU-only, no model assets)"
	@cd candle-binding && \
	test_list="$$(cargo test --release --no-default-features --lib -- --list)" && \
	for test_filter in $(RUST_CI_LIB_TESTS); do \
		echo "$$test_list" | grep -F "$${test_filter}:" >/dev/null || { \
			echo "Configured Rust CI test not found: $$test_filter"; \
			exit 1; \
		}; \
		echo "Running $$test_filter"; \
		cargo test --release --no-default-features --lib "$$test_filter" -- --exact --test-threads=1 --nocapture || exit 1; \
	done

# Test Rust unit tests (with release optimization for performance)
# Note: Uses TEST_GPU_DEVICE env var (default: 2) to avoid GPU 0/1 which may be busy
# Override with: TEST_GPU_DEVICE=3 make test-rust
test-rust: rust
	@$(LOG_TARGET)
	@echo "Running Rust unit tests (release mode, sequential on GPU $(TEST_GPU_DEVICE))"
	@cd candle-binding && CUDA_VISIBLE_DEVICES=$(TEST_GPU_DEVICE) cargo test --release --lib -- --test-threads=1 --nocapture

# Test Flash Attention（requires GPU and CUDA environment configured in system）
# Note: Ensure CUDA paths are set in your shell environment (e.g., ~/.bashrc)
#   - PATH should include nvcc (e.g., /usr/local/cuda/bin)
#   - LD_LIBRARY_PATH should include CUDA libs (e.g., /usr/local/cuda/lib64, /usr/lib/wsl/lib for WSL)
#   - CUDA_HOME, CUDA_PATH should point to CUDA installation
# Note: Uses TEST_GPU_DEVICE env var (default: 2) to avoid GPU 0/1 which may be busy
test-rust-flash-attn: rust-flash-attn
	@$(LOG_TARGET)
	@echo "Running Rust unit tests with Flash Attention 2 (GPU $(TEST_GPU_DEVICE))"
	@cd candle-binding && CUDA_VISIBLE_DEVICES=$(TEST_GPU_DEVICE) cargo test --release --features flash-attn --lib -- --test-threads=1 --nocapture

# Test specific Rust module (with release optimization for performance)
#   Example: make test-rust-module MODULE=classifiers::lora::pii_lora_test
#   Example: make test-rust-module MODULE=classifiers::lora::pii_lora_test::test_pii_lora_pii_lora_classifier_new
test-rust-module: rust
	@$(LOG_TARGET)
	@if [ -z "$(MODULE)" ]; then \
		echo "Usage: make test-rust-module MODULE=<module_name>"; \
		echo "Example: make test-rust-module MODULE=core::similarity_test"; \
		exit 1; \
	fi
	@echo "Running Rust tests for module: $(MODULE) (release mode, GPU $(TEST_GPU_DEVICE))"
	@cd candle-binding && CUDA_VISIBLE_DEVICES=$(TEST_GPU_DEVICE) cargo test --release $(MODULE) --lib -- --test-threads=1 --nocapture

# Test specific Flash Attention module (requires GPU and CUDA environment)
#   Example: make test-rust-flash-attn-module MODULE=model_architectures::embedding::qwen3_embedding_test
#   Example: make test-rust-flash-attn-module MODULE=model_architectures::embedding::qwen3_embedding_test::test_qwen3_embedding_forward
test-rust-flash-attn-module: rust-flash-attn
	@$(LOG_TARGET)
	@if [ -z "$(MODULE)" ]; then \
		echo "Usage: make test-rust-flash-attn-module MODULE=<module_name>"; \
		echo "Example: make test-rust-flash-attn-module MODULE=model_architectures::embedding::qwen3_embedding_test"; \
		exit 1; \
	fi
	@echo "Running Rust Flash Attention tests for module: $(MODULE) (GPU $(TEST_GPU_DEVICE))"
	@cd candle-binding && CUDA_VISIBLE_DEVICES=$(TEST_GPU_DEVICE) cargo test --release --features flash-attn $(MODULE) --lib -- --nocapture

# Test the Rust library - minimal models only (conditionally use rust-ci in CI environments)
# The MultiModal entries are the hermetic (no-network) subset of the
# MULTIMODAL_MODEL_PATH-gated tests; they skip cleanly when the variable is
# unset and run against models/mom-embedding-multimodal in CI, where
# download-models has already fetched it (issue #2319). The image-encode
# binding tests (TestMultiModalEncodeImageFrom*) stay out of this lane because
# they download fixture images from Wikimedia at test time; run them via
# make test-binding-multimodal.
test-binding-minimal: $(if $(CI),rust-ci,rust) ## Run Go tests with minimal models (BERT, ModernBERT)
	@$(LOG_TARGET)
	@echo "Running candle-binding tests with minimal models (BERT, ModernBERT classifiers)..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race \
		-run "^Test(InitModel|Tokenization|Embeddings|Similarity|FindMostSimilar|ModernBERTClassifiers|ModernBertClassifier_ConcurrentClassificationSafety|ModernBERTPIITokenClassification|UtilityFunctions|ErrorHandling|Concurrency|MultiModalEmbeddingInit|MultiModalEncodeText|MultiModalInputValidation)$$"

# Run every MULTIMODAL_MODEL_PATH-gated test against a local model copy:
# the candle-binding Go tests (including the network-dependent image-encode
# ones), the Go router integration tests in pkg/classification, and the
# ignored Rust unit tests in multimodal_embedding.rs. This is the manual
# receipt command for PRs touching the multimodal FFI (issue #2319).
# Requires models/mom-embedding-multimodal (make download-models) and network
# access for the Wikimedia fixture images.
test-binding-multimodal: $(if $(CI),rust-ci,rust) ## Run the multimodal model-gated Go tests (binding + router integration)
	@$(LOG_TARGET)
	@if [ ! -d "$${MULTIMODAL_MODEL_PATH:-$(CURDIR)/models/mom-embedding-multimodal}" ]; then \
		echo "Multimodal model not found at $${MULTIMODAL_MODEL_PATH:-$(CURDIR)/models/mom-embedding-multimodal}"; \
		echo "Run 'make download-models' first, or set MULTIMODAL_MODEL_PATH."; \
		exit 1; \
	fi
	@echo "Running candle-binding multimodal Go tests..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
		export MULTIMODAL_MODEL_PATH=$${MULTIMODAL_MODEL_PATH:-$(CURDIR)/models/mom-embedding-multimodal} && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race -run "^TestMultiModal" .
	@echo "Running Go router multimodal integration tests (pkg/classification)..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
		export MULTIMODAL_MODEL_PATH=$${MULTIMODAL_MODEL_PATH:-$(CURDIR)/models/mom-embedding-multimodal} && \
		cd src/semantic-router && CGO_ENABLED=1 \
		CGO_LDFLAGS="-L$(CURDIR)/candle-binding/target/release" \
		go test -v -run "^TestEmbeddingClassifier_Integration" ./pkg/classification/

# Exploratory lane for the #[ignore] Rust multimodal unit tests. Kept OUT of
# test-binding-multimodal so that target stays a pass/fail receipt: this suite
# has a known-red baseline (see docs/agent/testing-strategy.md, "Model-Gated
# Multimodal Tests") and is expected to exit non-zero until those pre-existing
# defects are fixed.
test-binding-multimodal-rust-baseline: $(if $(CI),rust-ci,rust) ## Run the ignored Rust multimodal unit tests (known-red baseline)
	@$(LOG_TARGET)
	@echo "Running ignored Rust multimodal unit tests (known-red baseline; see docs/agent/testing-strategy.md)..."
	@cd candle-binding && \
		MULTIMODAL_MODEL_PATH=$${MULTIMODAL_MODEL_PATH:-$(CURDIR)/models/mom-embedding-multimodal} \
		cargo test --release --no-default-features --lib multimodal_embedding::integration_tests -- --ignored --test-threads=1

# Test the Rust library - LoRA and advanced embedding models (conditionally use rust-ci in CI environments)
test-binding-lora: $(if $(CI),rust-ci,rust) ## Run Go tests with LoRA and advanced embedding models
	@$(LOG_TARGET)
	@echo "Running candle-binding tests with LoRA and advanced embedding models..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race \
		-run "^Test(BertTokenClassification|BertSequenceClassification|CandleBertClassifier|CandleBertTokenClassifier|CandleBertTokensWithLabels|LoRAUnifiedClassifier|GetEmbeddingSmart|InitEmbeddingModels|GetEmbeddingWithDim|EmbeddingConsistency|EmbeddingPriorityRouting|EmbeddingConcurrency)$$" \
		|| { echo "⚠️  Warning: Some LoRA/embedding tests failed (may be due to missing restricted models), continuing..."; $(if $(CI),true,exit 1); }
# Test the Rust library - all tests (conditionally use rust-ci in CI environments)
test-binding: $(if $(CI),rust-ci,rust) ## Run all Go tests with the Rust static library
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race

# Test with the candle-binding library (conditionally use rust-ci in CI environments)
test-category-classifier: $(if $(CI),rust-ci,rust) ## Test domain classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
		cd src/training/classifier_model_fine_tuning && CGO_ENABLED=1 go run test_linear_classifier.go

# Test the PII classifier (conditionally use rust-ci in CI environments)
test-pii-classifier: $(if $(CI),rust-ci,rust) ## Test PII classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
		cd src/training/pii_model_fine_tuning && CGO_ENABLED=1 go run pii_classifier_verifier.go

# Test the jailbreak classifier (conditionally use rust-ci in CI environments)
test-jailbreak-classifier: $(if $(CI),rust-ci,rust) ## Test jailbreak classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release:${PWD}/nlp-binding/target/release && \
		cd src/training/prompt_guard_fine_tuning && CGO_ENABLED=1 go run jailbreak_classifier_verifier.go

# Build the Rust library (with CUDA by default, Flash Attention optional)
# Set ENABLE_FLASH_ATTN=1 to enable Flash Attention: make rust ENABLE_FLASH_ATTN=1
rust: ## Ensure Rust is installed and build the Rust library with CUDA support (Flash Attention optional via ENABLE_FLASH_ATTN=1)
	@$(LOG_TARGET)
	@bash -c 'if ! command -v rustc >/dev/null 2>&1; then \
		echo "rustc not found, installing..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
	fi && \
	if [ -f "$$HOME/.cargo/env" ]; then \
		echo "Loading Rust environment from $$HOME/.cargo/env..." && \
		. $$HOME/.cargo/env; \
	fi && \
	if ! command -v cargo >/dev/null 2>&1; then \
		echo "Error: cargo not found in PATH" && exit 1; \
	fi && \
	if [ "$$ENABLE_FLASH_ATTN" = "1" ]; then \
		if command -v nvcc >/dev/null 2>&1; then \
			echo "Building Rust library with CUDA and Flash Attention support (ENABLE_FLASH_ATTN=1)..." && \
			echo "nvcc found: $$(nvcc --version | grep release)" && \
			echo "   Note: Flash Attention requires CUDA Compute Capability >= 8.0 (RTX 3090+, A100, H100)" && \
			cd candle-binding && cargo build --release --features flash-attn; \
		else \
			echo "❌ Error: ENABLE_FLASH_ATTN=1 but nvcc not found" && \
			echo "   Flash Attention requires CUDA environment. Install CUDA toolkit or unset ENABLE_FLASH_ATTN." && \
			exit 1; \
		fi; \
	else \
		if command -v nvcc >/dev/null 2>&1; then \
			echo "Building Rust library with CUDA support..." && \
			echo "💡 Tip: For 20-30% speedup on RTX 3090+/A100/H100, use: make rust ENABLE_FLASH_ATTN=1" && \
			cd candle-binding && cargo build --release; \
		else \
			echo "Building Rust library for CPU (nvcc not found)..." && \
			cd candle-binding && cargo build --release --no-default-features; \
		fi; \
	fi && \
	echo "Building ml-binding Rust library..." && \
	cd ../ml-binding && cargo build --release && \
	echo "Building nlp-binding Rust library..." && \
	cd ../nlp-binding && \
	rm -f target/release/libnlp_binding.dylib target/release/deps/libnlp_binding.dylib \
		target/release/libnlp_binding.so target/release/deps/libnlp_binding.so && \
	cargo build --release'

# Build the Rust library without CUDA (for CI/CD environments)
rust-ci: ## Build the Rust library without CUDA support (for GitHub Actions/CI)
	@$(LOG_TARGET)
	@bash -c 'if ! command -v rustc >/dev/null 2>&1; then \
		echo "rustc not found, installing..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
	fi && \
	if [ -f "$$HOME/.cargo/env" ]; then \
		echo "Loading Rust environment from $$HOME/.cargo/env..." && \
		. $$HOME/.cargo/env; \
	fi && \
	if ! command -v cargo >/dev/null 2>&1; then \
		echo "Error: cargo not found in PATH" && exit 1; \
	fi && \
	echo "Building Rust library without CUDA (CPU-only)..." && \
	cd candle-binding && cargo build --release --no-default-features && \
	echo "Building ml-binding Rust library..." && \
	cd ../ml-binding && cargo build --release && \
	echo "Building nlp-binding Rust library..." && \
	cd ../nlp-binding && \
	rm -f target/release/libnlp_binding.dylib target/release/deps/libnlp_binding.dylib \
		target/release/libnlp_binding.so target/release/deps/libnlp_binding.so && \
	cargo build --release'

rust-flash-attn: ## Build Rust library with Flash Attention 2 (requires CUDA environment)
	@$(LOG_TARGET)
	@echo "Building Rust library with Flash Attention 2 (requires CUDA)..."
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "nvcc found: $$(nvcc --version | grep release)"; \
	else \
		echo "❌ nvcc not found in PATH. Please configure CUDA environment."; \
		exit 1; \
	fi
	@cd candle-binding && cargo build --release --features flash-attn
	@echo "Building ml-binding Rust library..."
	@cd ml-binding && cargo build --release
	@echo "Building nlp-binding Rust library..."
	@cd nlp-binding && rm -f target/release/libnlp_binding.dylib target/release/deps/libnlp_binding.dylib \
		target/release/libnlp_binding.so target/release/deps/libnlp_binding.so && cargo build --release
