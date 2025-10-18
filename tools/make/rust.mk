# ======== rust.mk ========
# = Everything For rust   =
# ======== rust.mk ========

##@ Rust

# Test the Rust library
test-binding: rust ## Run Go tests with the Rust static library
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race

# Test with the candle-binding library
test-category-classifier: rust ## Test domain classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd src/training/classifier_model_fine_tuning && CGO_ENABLED=1 go run test_linear_classifier.go

# Test the PII classifier
test-pii-classifier: rust ## Test PII classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd src/training/pii_model_fine_tuning && CGO_ENABLED=1 go run pii_classifier_verifier.go

# Test the jailbreak classifier
test-jailbreak-classifier: rust ## Test jailbreak classifier with candle-binding
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd src/training/prompt_guard_fine_tuning && CGO_ENABLED=1 go run jailbreak_classifier_verifier.go

# Build the Rust library
rust: ## Ensure Rust is installed and build the Rust library
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
	echo "Building Rust library..." && \
	cd candle-binding && cargo build --release'
