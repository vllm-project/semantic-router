# ======== rust.mk ========
# = Everything For rust   =
# ======== rust.mk ========

# Test the Rust library
test-binding: rust
	@$(LOG_TARGET)
	@echo "Running Go tests with static library..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race

# Test with the candle-binding library
test-category-classifier: rust
	@$(LOG_TARGET)
	@echo "Testing domain classifier with candle-binding..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd src/training/classifier_model_fine_tuning && CGO_ENABLED=1 go run test_linear_classifier.go

# Test the PII classifier
test-pii-classifier: rust
	@$(LOG_TARGET)
	@echo "Testing PII classifier with candle-binding..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd src/training/pii_model_fine_tuning && CGO_ENABLED=1 go run pii_classifier_verifier.go

# Test the jailbreak classifier
test-jailbreak-classifier: rust
	@$(LOG_TARGET)
	@echo "Testing jailbreak classifier with candle-binding..."
	@export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release && \
		cd src/training/prompt_guard_fine_tuning && CGO_ENABLED=1 go run jailbreak_classifier_verifier.go

# Build the Rust library
rust:
	@$(LOG_TARGET)
	@echo "Ensuring rust is installed..."
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
