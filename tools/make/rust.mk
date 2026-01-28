# ======== rust.mk ========
# = Everything For rust   =
# ======== rust.mk ========

# Default GPU device for testing (can be overridden: TEST_GPU_DEVICE=3 make test-rust)
TEST_GPU_DEVICE ?= 2

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
test-binding-minimal: $(if $(CI),rust-ci,rust) ## Run Go tests with minimal models (BERT, ModernBERT)
	@$(LOG_TARGET)
	@echo "Running candle-binding tests with minimal models (BERT, ModernBERT classifiers)..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race \
		-run "^Test(InitModel|Tokenization|Embeddings|Similarity|FindMostSimilar|ModernBERTClassifiers|ModernBertClassifier_ConcurrentClassificationSafety|ModernBERTPIITokenClassification|UtilityFunctions|ErrorHandling|Concurrency)$$"

# Test the Rust library - LoRA and advanced embedding models (conditionally use rust-ci in CI environments)
test-binding-lora: $(if $(CI),rust-ci,rust) ## Run Go tests with LoRA and advanced embedding models
	@$(LOG_TARGET)
	@echo "Running candle-binding tests with LoRA and advanced embedding models..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race \
		-run "^Test(BertTokenClassification|BertSequenceClassification|CandleBertClassifier|CandleBertTokenClassifier|CandleBertTokensWithLabels|LoRAUnifiedClassifier|GetEmbeddingSmart|InitEmbeddingModels|GetEmbeddingWithDim|EmbeddingConsistency|EmbeddingPriorityRouting|EmbeddingConcurrency)$$" \
		|| { echo "⚠️  Warning: Some LoRA/embedding tests failed (may be due to missing restricted models), continuing..."; $(if $(CI),true,exit 1); }
# Test the Rust library - all tests (conditionally use rust-ci in CI environments)
test-binding: $(if $(CI),rust-ci,rust) ## Run all Go tests with the Rust static library
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race

# Test with the candle-binding library (conditionally use rust-ci in CI environments)
test-category-classifier: $(if $(CI),rust-ci,rust) ## Test domain classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release && \
		cd src/training/classifier_model_fine_tuning && CGO_ENABLED=1 go run test_linear_classifier.go

# Test the PII classifier (conditionally use rust-ci in CI environments)
test-pii-classifier: $(if $(CI),rust-ci,rust) ## Test PII classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release && \
		cd src/training/pii_model_fine_tuning && CGO_ENABLED=1 go run pii_classifier_verifier.go

# Test the jailbreak classifier (conditionally use rust-ci in CI environments)
test-jailbreak-classifier: $(if $(CI),rust-ci,rust) ## Test jailbreak classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release && \
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
			echo "✅ nvcc found: $$(nvcc --version | grep release)" && \
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
	fi'

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
	cd ../ml-binding && cargo build --release'

rust-flash-attn: ## Build Rust library with Flash Attention 2 (requires CUDA environment)
	@$(LOG_TARGET)
	@echo "Building Rust library with Flash Attention 2 (requires CUDA)..."
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "✅ nvcc found: $$(nvcc --version | grep release)"; \
	else \
		echo "❌ nvcc not found in PATH. Please configure CUDA environment."; \
		exit 1; \
	fi
	@cd candle-binding && cargo build --release --features flash-attn
