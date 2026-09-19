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
	model_architectures::model_factory::tokenizer_contract_tests::mmbert_embedding_discards_saved_training_limits \
	core::tokenization_test::test_tokenization_config_default \
	model_architectures::embedding::pooling_test::test_mean_pool_long_low_precision \
	model_architectures::embedding::pooling_test::test_mean_pool_padding_and_invalid_rows \
	model_architectures::traditional::modernbert::head_contract_tests::head_honors_configured_epsilon_and_rejects_partial_weights \
	model_architectures::embedding::representation_contract::tests::honors_hf_default_and_rejects_unknown_representation \
	model_architectures::embedding::mmbert_embedding::early_exit_contract_tests::intermediate_embeddings_preserve_hf_hidden_state_contract \
	model_architectures::embedding::mmbert_embedding::early_exit_contract_tests::long_context_rotary_preserves_positions_before_half_cast \
	core::tokenization_window::tests::test_window_ranges_cover_every_token \
	core::tokenization_window::tests::test_window_ranges_overlap_on_a_short_stride \
	core::tokenization_window::tests::test_window_ranges_edges \
	core::tokenization_test::test_tokenization_config_custom \
	ffi::embedding_test::test_truncate_embedding_renormalizes_prefix \
	ffi::capabilities::tests::normalizes_known_model_types \
	ffi::capabilities::tests::reports_multimodal_modalities \
	ffi::capabilities::tests::distinguishes_unsupported_and_invalid_input \
	ffi::capabilities::tests::observed_dimension_buffer_roundtrip \
	ffi::capability_dimensions::tests::preserves_declared_dimensions_and_native_width \
	ffi::capability_dimensions::tests::rejects_invalid_model_metadata \
	model_architectures::embedding::multimodal_embedding::tests::test_loaded_dimensions_follow_model_configuration \
	model_architectures::embedding::mmbert_embedding::tests::test_early_exit_preserves_residual_and_full_depth_applies_final_norm \
	model_architectures::embedding::multimodal_embedding::tests::test_siglip_vision_encoder_loads_with_head_weights \
	model_architectures::embedding::multimodal_embedding::tests::test_siglip_vision_encoder_requires_pooling_head \
	model_architectures::traditional::candle_models::modernbert::tests::test_chunked_attention_matches_dense \
	model_architectures::traditional::candle_models::modernbert::tests::test_chunked_attention_matches_dense_with_padding \
	model_architectures::traditional::candle_models::modernbert::tests::test_chunked_attention_crosses_default_context_and_query_blocks \
	model_architectures::traditional::candle_models::modernbert::tests::test_flash_attention_never_changes_requested_precision \
	model_architectures::traditional::candle_models::modernbert::tests::test_flash_option_keeps_fp32_cpu_attention_unchanged \
	model_architectures::traditional::modernbert_test::test_candle_context_default_and_explicit_limits \
	model_architectures::traditional::modernbert_test::test_candle_context_budget_reserves_actual_postprocessor_special_tokens \
	model_architectures::traditional::modernbert_test::test_candle_context_tokenization_preserves_tail_and_model_padding \
	model_architectures::traditional::modernbert_test::test_candle_context_config_preserves_separate_rope_theta \
	model_architectures::traditional::modernbert_test::test_candle_context_loaders_reject_invalid_limits_before_weights \
	model_architectures::traditional::modernbert_test::test_candle_context_classifier_loaders_execute_beyond_default \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_single_query_over_many_keys \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_matches_dense_with_decode_offset \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_offset_prefill_equals_split_prefill \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_rejects_a_block_with_no_keys \
	model_architectures::attention::chunked_sdpa_test::test_cpu_softmax_matches_original_boundaries \
	model_architectures::attention::chunked_sdpa_test::test_cpu_softmax_preserves_partial_and_all_padding \
	model_architectures::attention::chunked_sdpa_test::test_cpu_softmax_preserves_all_negative_infinity \
	model_architectures::attention::chunked_sdpa_test::test_cpu_softmax_matches_dense_across_key_lengths \
	model_architectures::attention::chunked_sdpa_test::test_cpu_softmax_matches_dense_with_cached_keys \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_key_blocks_match_dense \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_key_blocks_match_dense_with_decode_offset \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_all_masked_first_key_tile_stays_finite \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_single_query_with_masked_first_keys_stays_finite \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_f64_scores_below_f32_min_stay_finite \
	model_architectures::attention::chunked_sdpa_test::test_chunked_sdpa_half_precision_accumulates_without_overflow \
	model_architectures::embedding::gemma3_model::chunked_attention_tests::test_chunked_attention_matches_dense \
	model_architectures::embedding::qwen3_embedding::chunked_attention_tests::test_chunked_attention_matches_dense \
	model_architectures::embedding::qwen3_embedding::chunked_attention_tests::test_chunked_attention_matches_dense_on_real_rows_with_left_padding \
	model_architectures::embedding::multimodal_embedding::tests::test_bert_self_attention_matches_dense \
	model_architectures::embedding::multimodal_embedding::tests::test_siglip_and_whisper_self_attention_match_dense \
	model_architectures::embedding::multimodal_embedding::tests::test_siglip_head_attention_matches_dense \
	model_architectures::generative::qwen3_with_lora::chunked_attention_tests::test_prefill_and_decode_match_dense \
	model_architectures::generative::qwen3_with_lora::chunked_attention_tests::test_cached_suffix_generation_matches_uncached \
	model_architectures::traditional::base_model_test::test_self_attention_matches_dense_reference

RUST_CI_LIB_TEST_GROUPS ?= ffi::instances::tests:: core::sequence_windows::tests:: core::token_windows::tests::

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
	done && \
	for test_filter in $(RUST_CI_LIB_TEST_GROUPS); do \
		echo "$$test_list" | grep -F "$$test_filter" >/dev/null || { \
			echo "Configured Rust CI test group not found: $$test_filter"; \
			exit 1; \
		}; \
		cargo test --release --no-default-features --lib "$$test_filter" -- --test-threads=1 --nocapture || exit 1; \
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

# Hermetic Go/C ABI contracts. Published checkpoint inference is a separate
# required suite (make test-models), not a whitelist of optional legacy tests.
# The same contracts run under RISC-V QEMU without the unsupported race detector.
BINDING_MINIMAL_GO_TESTS ?= ^Test(Owned.*|NewRegexProvider|RegexProvider_.*|UtilityFunctions|EmbeddingCapabilitiesConformance|EmbeddingDimensionStateValidation)$$
# This checkpoint case belongs to legacy-hallucination-checkpoints in core_test_profiles.json.
BINDING_MINIMAL_GO_SKIP ?= ^TestOwnedNativeMaintainedHallucinationWithoutLabelMetadata$$

test-binding-minimal: $(if $(CI),rust-ci,rust) ## Run model-free owned native binding contracts
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race -count=1 \
		-run '$(BINDING_MINIMAL_GO_TESTS)' -skip '$(BINDING_MINIMAL_GO_SKIP)' .

# Tiny checked-in/generated tensors exercise the actual native libraries without
# downloading checkpoints. CI uses the same API22 CPU runtime as the CPU image;
# an explicit ORT_DYLIB_PATH selects an already installed compatible runtime.
OWNED_TEST_ORT_VERSION ?= 1.22.0
test-owned-native: $(if $(CI),rust-ci,rust) harness-venv-install ## Test owned native instances and classification assembly with real CPU tensors
	@if [ -z "$${ORT_DYLIB_PATH:-}" ]; then \
		"$(AGENT_PYTHON)" -m pip install --quiet "onnxruntime==$(OWNED_TEST_ORT_VERSION)"; \
	fi
	@set -e; \
	if [ -z "$${ORT_DYLIB_PATH:-}" ]; then \
		ORT_DYLIB_PATH="$$("$(AGENT_PYTHON)" -c 'import importlib.util, pathlib; root = pathlib.Path(importlib.util.find_spec("onnxruntime").origin).parent / "capi"; paths = list(root.glob("libonnxruntime.so.*")) + list(root.glob("libonnxruntime.*.dylib")); assert len(paths) == 1, paths; print(paths[0])')"; \
		export ORT_DYLIB_PATH; \
	fi; \
	export $(NATIVE_ENV); \
	cd onnx-binding; \
	CGO_ENABLED=1 go test -race -count=1 ./instance; \
	cd ../src/semantic-router; \
	CORE_NATIVE_FIXTURES=1 CGO_ENABLED=1 go test -race -count=1 ./pkg/classification \
		-run '^(TestNativeMappingCandidateKeepsPreviousModel|TestTwoLocalRulesOwnNativeModelsInOneRecipe|TestLegacyStartupUsesProjectedNativeMapping)$$'; \
	CGO_ENABLED=1 go test -race -count=1 ./pkg/modelruntime/native \
		-run '^(TestORTEmbeddingPreparesEveryAdvertisedLayerBeforePublication|TestORTOwnedEncoder32KOverflowPolicies)$$'; \
	CGO_ENABLED=1 go test -race -count=1 ./pkg/modelruntime ./pkg/modeldownload \
		-ldflags='-X github.com/vllm-project/semantic-router/src/semantic-router/pkg/config.defaultModelProvider=ort' \
		-run '^(TestOwnedImplicitORTEmbeddingAndExplicitCandleOverride|TestImplicitEmbeddingProvisioningFollowsBuildProvider)$$'

# The CK flash-attention graph rewriter is a Python script under onnx-binding;
# Its tests also execute blocked FP32 graphs with the CPU runtime.
CK_REWRITE_SCRIPTS_DIR ?= onnx-binding/ort-ck-flash-attn/scripts
CK_REWRITE_PYTHON_DEPS ?= onnx==1.22.0 onnxruntime==1.24.2

ck-rewrite-deps: harness-venv-install ## Install the CK graph rewriter test dependencies into the harness venv
	@"$(AGENT_PYTHON)" -c "import onnx, onnxruntime" 2>/dev/null || "$(AGENT_PYTHON)" -m pip install --quiet $(CK_REWRITE_PYTHON_DEPS)

ck-rewrite-test: ck-rewrite-deps ## Run the CK flash-attention graph rewriter unit tests
	@$(LOG_TARGET)
	@cd $(CK_REWRITE_SCRIPTS_DIR) && "$(AGENT_PYTHON)" -m unittest test_rewrite_graph test_stable_pooling test_rewrite_blocked_attention test_canonicalize_attention_masks test_reshape_dimensions

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
	@export $(NATIVE_ENV) && \
		export MULTIMODAL_MODEL_PATH=$${MULTIMODAL_MODEL_PATH:-$(CURDIR)/models/mom-embedding-multimodal} && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race -run "^TestMultiModal" .
	@echo "Running Go router multimodal integration tests (pkg/classification)..."
	@export $(NATIVE_ENV) && \
		export MULTIMODAL_MODEL_PATH=$${MULTIMODAL_MODEL_PATH:-$(CURDIR)/models/mom-embedding-multimodal} && \
		cd src/semantic-router && CGO_ENABLED=1 \
		go test -v -run "^TestEmbeddingClassifier_Integration" ./pkg/classification/

# Exploratory lane for the #[ignore] Rust multimodal unit tests. Kept OUT of
# test-binding-multimodal so that target stays a pass/fail receipt: this suite
# has a known-red baseline (see tools/agent/docs/testing-strategy.md, "Model-Gated
# Multimodal Tests") and is expected to exit non-zero until those pre-existing
# defects are fixed.
test-binding-multimodal-rust-baseline: $(if $(CI),rust-ci,rust) ## Run the ignored Rust multimodal unit tests (known-red baseline)
	@$(LOG_TARGET)
	@echo "Running ignored Rust multimodal unit tests (known-red baseline; see tools/agent/docs/testing-strategy.md)..."
	@cd candle-binding && \
		MULTIMODAL_MODEL_PATH=$${MULTIMODAL_MODEL_PATH:-$(CURDIR)/models/mom-embedding-multimodal} \
		cargo test --release --no-default-features --lib multimodal_embedding::integration_tests -- --ignored --test-threads=1

# Test the Rust library - LoRA and advanced embedding models (conditionally use rust-ci in CI environments)
test-binding-lora: $(if $(CI),rust-ci,rust) ## Run Go tests with LoRA and advanced embedding models
	@$(LOG_TARGET)
	@echo "Running candle-binding tests with LoRA and advanced embedding models..."
	@export $(NATIVE_ENV) && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race \
		-run "^Test(BertTokenClassification|BertSequenceClassification|CandleBertClassifier|CandleBertTokenClassifier|CandleBertTokensWithLabels|LoRAUnifiedClassifier|GetEmbeddingSmart|InitEmbeddingModels|GetEmbeddingWithDim|EmbeddingConsistency|EmbeddingPriorityRouting|EmbeddingConcurrency)$$"
# Test the Rust library - all tests (conditionally use rust-ci in CI environments)
test-binding: $(if $(CI),rust-ci,rust) ## Run all Go tests with the Rust static library
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
		cd candle-binding && CGO_ENABLED=1 go test -v -race

# Test with the candle-binding library (conditionally use rust-ci in CI environments)
test-category-classifier: $(if $(CI),rust-ci,rust) ## Test domain classifier with candle-binding
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
		cd src/training/classifier_model_fine_tuning && CGO_ENABLED=1 go run test_linear_classifier.go

# Test the PII classifier (conditionally use rust-ci in CI environments)
test-pii-classifier: $(if $(CI),rust-ci,rust) ## Test PII classifier with candle-binding
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
		cd src/training/pii_model_fine_tuning && CGO_ENABLED=1 go run pii_classifier_verifier.go

# Test the jailbreak classifier (conditionally use rust-ci in CI environments)
test-jailbreak-classifier: $(if $(CI),rust-ci,rust) ## Test jailbreak classifier with candle-binding
	@$(LOG_TARGET)
	@export $(NATIVE_ENV) && \
		cd src/training/prompt_guard_fine_tuning && CGO_ENABLED=1 go run jailbreak_classifier_verifier.go

# Build the Rust library (with CUDA by default, Flash Attention optional)
# Set ENABLE_FLASH_ATTN=1 to enable Flash Attention: make rust ENABLE_FLASH_ATTN=1
rust: build-onnx-binding
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
ifeq ($(PREBUILT_NATIVE_LIBS),1)
rust-ci: ## Verify shared native libraries instead of rebuilding them
	@python3 tools/ci/native_artifact.py verify --directory "$(NATIVE_ARTIFACT_DIR)"
else
rust-ci: build-onnx-binding
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
endif

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

# Cross-compile Candle CPU classifiers for riscv64 and run them under qemu-user.
# This is ISA smoke, not hardware qualification. Go coverage matches
# test-binding-minimal except -race (unsupported on linux/riscv64).
# Provisions only the registry-pinned Vela Domain checkpoint and records the
# same immutable artifact identity used by the other published-model suites.
# After binding tests, build-router-riscv links a Candle-only process and the
# QEMU smoke hits /health plus one classify/intent call.
RISCV_GNU_TARGET ?= riscv64gc-unknown-linux-gnu
RISCV_GNU_CC ?= riscv64-linux-gnu-gcc
RISCV_GNU_CXX ?= riscv64-linux-gnu-g++
RISCV_SYSROOT ?= /usr/riscv64-linux-gnu
RISCV_QEMU ?= $(firstword $(wildcard /usr/bin/qemu-riscv64-static /usr/bin/qemu-riscv64))
RISCV_QEMU_TEST ?= $(CURDIR)/candle-binding/target/$(RISCV_GNU_TARGET)/candle-riscv64.test
RISCV_MODEL_MANIFEST ?= $(MODEL_TEST_REPORT_DIR)/models.json
RISCV_CLASSIFIER_MODEL ?= $(shell python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["models"][0]["path"])' "$(RISCV_MODEL_MANIFEST)")
RISCV_CLASSIFIER_PARITY_GOLDEN ?= $(CURDIR)/candle-binding/target/$(RISCV_GNU_TARGET)/classifier-parity.json
RISCV_CANDLE_LIBDIR ?= $(CURDIR)/candle-binding/target/$(RISCV_GNU_TARGET)/release
RISCV_ROUTER_BIN ?= $(CURDIR)/bin/router-riscv64
RISCV_ROUTER_CONFIG ?= $(CURDIR)/e2e/config/config.riscv-qemu.yaml
RISCV_ROUTER_API_PORT ?= 18080

download-riscv-classifier: ## Download only the Vela Domain checkpoint used by test-riscv-qemu
	@$(LOG_TARGET)
	@cd src/semantic-router && go run ./tools/model-test-assets \
		--provider candle --suite riscv --output "$(MODEL_TEST_MODELS_DIR)" \
		--manifest "$(RISCV_MODEL_MANIFEST)" --download

RISCV_QEMU_LIB_TESTS ?= \
	model_architectures::traditional::modernbert_test::test_candle_context_classifier_loaders_execute_beyond_default \
	model_architectures::traditional::candle_models::modernbert::tests::test_chunked_attention_matches_dense

test-riscv-qemu: download-riscv-classifier ## Cross-compile Candle CPU classifiers and smoke the Candle-only router under qemu-user riscv64
	@$(LOG_TARGET)
	@mkdir -p "$(MODEL_TEST_REPORT_DIR)"
	@printf '%s\n' $(RISCV_QEMU_LIB_TESTS) >"$(MODEL_TEST_REPORT_DIR)/rust-required.txt"
	@printf '%s\n' '$(BINDING_MINIMAL_GO_TESTS)' >"$(MODEL_TEST_REPORT_DIR)/minimal-pattern.txt"
	@printf '%s\n' '$(BINDING_MINIMAL_GO_SKIP)' >"$(MODEL_TEST_REPORT_DIR)/minimal-skip.txt"
	@if ! command -v $(RISCV_GNU_CC) >/dev/null 2>&1; then \
		echo "missing $(RISCV_GNU_CC); install gcc-riscv64-linux-gnu"; \
		exit 1; \
	fi
	@if [ -z "$(RISCV_QEMU)" ]; then \
		echo "missing qemu-riscv64; install qemu-user-static"; \
		exit 1; \
	fi
	@if [ ! -d "$(RISCV_SYSROOT)" ]; then \
		echo "missing RISC-V sysroot $(RISCV_SYSROOT); install libc6-dev-riscv64-cross"; \
		exit 1; \
	fi
	@if [ ! -f "$(RISCV_CLASSIFIER_MODEL)/config.json" ]; then \
		echo "missing $(RISCV_CLASSIFIER_MODEL); run make download-riscv-classifier"; \
		exit 1; \
	fi
	@echo "Building host Candle CPU library for amd64/arm64 parity snapshot"
	@cd candle-binding && cargo build --release --no-default-features
	@echo "Recording host classifier outputs for RISC-V parity"
	@export $(NATIVE_ENV) && \
		cd candle-binding && CGO_ENABLED=1 \
		CANDLE_CLASSIFIER_MODEL="$(RISCV_CLASSIFIER_MODEL)" \
		CANDLE_CLASSIFIER_PARITY_MODE=record \
		CANDLE_CLASSIFIER_PARITY_GOLDEN="$(RISCV_CLASSIFIER_PARITY_GOLDEN)" \
		go test -json -count=1 -timeout 30m -run '^TestCandleClassifierParity$$' \
		>"$(MODEL_TEST_REPORT_DIR)/host-parity.jsonl" 2>&1 || { \
			cat "$(MODEL_TEST_REPORT_DIR)/host-parity.jsonl"; exit 1; \
		}
	@cat "$(MODEL_TEST_REPORT_DIR)/host-parity.jsonl"
	@rustup target add $(RISCV_GNU_TARGET)
	@echo "Building Candle CPU library for $(RISCV_GNU_TARGET)"
	@cd candle-binding && \
		CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=$(RISCV_GNU_CC) \
		CC_riscv64gc_unknown_linux_gnu=$(RISCV_GNU_CC) \
		CXX_riscv64gc_unknown_linux_gnu=$(RISCV_GNU_CXX) \
		cargo build --release --no-default-features --target $(RISCV_GNU_TARGET)
	@echo "Running synthetic classifier and attention tests under qemu-user"
	@cd candle-binding && \
		test_list="$$(CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=$(RISCV_GNU_CC) \
			CC_riscv64gc_unknown_linux_gnu=$(RISCV_GNU_CC) \
			CXX_riscv64gc_unknown_linux_gnu=$(RISCV_GNU_CXX) \
			CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="$(RISCV_QEMU) -L $(RISCV_SYSROOT)" \
			cargo test --release --no-default-features --target $(RISCV_GNU_TARGET) --lib -- --list)" && \
		printf '%s\n' "$$test_list" >"$(MODEL_TEST_REPORT_DIR)/rust-list.txt" && \
		index=0 && \
		for test_filter in $(RISCV_QEMU_LIB_TESTS); do \
			echo "$$test_list" | grep -F "$${test_filter}:" >/dev/null || { \
				echo "Configured RISC-V QEMU test not found: $$test_filter"; \
				exit 1; \
			}; \
			echo "Running $$test_filter"; \
			status=0; \
			CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=$(RISCV_GNU_CC) \
				CC_riscv64gc_unknown_linux_gnu=$(RISCV_GNU_CC) \
				CXX_riscv64gc_unknown_linux_gnu=$(RISCV_GNU_CXX) \
				CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER="$(RISCV_QEMU) -L $(RISCV_SYSROOT)" \
				cargo test --release --no-default-features --target $(RISCV_GNU_TARGET) --lib "$$test_filter" -- --exact --test-threads=1 --nocapture \
				>"$(MODEL_TEST_REPORT_DIR)/rust-$$index.log" 2>&1 || status=$$?; \
			cat "$(MODEL_TEST_REPORT_DIR)/rust-$$index.log"; \
			[ "$$status" -eq 0 ] || exit "$$status"; \
			index=$$((index + 1)); \
		done
	@echo "Linking Go Candle FFI for linux/riscv64"
	@cd candle-binding && \
		CGO_ENABLED=1 GOOS=linux GOARCH=riscv64 CC=$(RISCV_GNU_CC) CXX=$(RISCV_GNU_CXX) \
		go test -c -o "$(RISCV_QEMU_TEST)" .
	@echo "Proving the riscv64 binary linked Candle instead of the unavailable stub"
	@cd candle-binding || exit; status=0; \
		LD_LIBRARY_PATH="$(CURDIR)/candle-binding/target/$(RISCV_GNU_TARGET)/release" \
		$(RISCV_QEMU) -L $(RISCV_SYSROOT) "$(RISCV_QEMU_TEST)" \
		-test.run '^TestNativeClassifierFFIIsLinked$$' -test.v=test2json -test.count=1 -test.timeout 10m \
		>"$(MODEL_TEST_REPORT_DIR)/ffi.log" 2>&1 || status=$$?; \
		cat "$(MODEL_TEST_REPORT_DIR)/ffi.log"; \
		go tool test2json -p candle-binding -t <"$(MODEL_TEST_REPORT_DIR)/ffi.log" >"$(MODEL_TEST_REPORT_DIR)/qemu-ffi.jsonl" && \
		[ "$$status" -eq 0 ]
	@echo "Comparing RISC-V classifier outputs against the host snapshot"
	@cd candle-binding || exit; status=0; \
		CANDLE_CLASSIFIER_MODEL="$(RISCV_CLASSIFIER_MODEL)" \
		CANDLE_CLASSIFIER_PARITY_MODE=compare \
		CANDLE_CLASSIFIER_PARITY_GOLDEN="$(RISCV_CLASSIFIER_PARITY_GOLDEN)" \
		LD_LIBRARY_PATH="$(CURDIR)/candle-binding/target/$(RISCV_GNU_TARGET)/release" \
		$(RISCV_QEMU) -L $(RISCV_SYSROOT) "$(RISCV_QEMU_TEST)" \
		-test.run '^TestCandleClassifierParity$$' -test.v=test2json -test.count=1 -test.timeout 60m \
		>"$(MODEL_TEST_REPORT_DIR)/parity.log" 2>&1 || status=$$?; \
		cat "$(MODEL_TEST_REPORT_DIR)/parity.log"; \
		go tool test2json -p candle-binding -t <"$(MODEL_TEST_REPORT_DIR)/parity.log" >"$(MODEL_TEST_REPORT_DIR)/qemu-parity.jsonl" && \
		[ "$$status" -eq 0 ]
	@echo "Running test-binding-minimal Go cases under qemu-user (no -race)"
	@cd candle-binding && \
		LD_LIBRARY_PATH="$(CURDIR)/candle-binding/target/$(RISCV_GNU_TARGET)/release" \
		$(RISCV_QEMU) -L $(RISCV_SYSROOT) "$(RISCV_QEMU_TEST)" \
		-test.list '$(BINDING_MINIMAL_GO_TESTS)' >"$(MODEL_TEST_REPORT_DIR)/binding-list.txt"
	@cd candle-binding || exit; status=0; \
		QEMU_LD_PREFIX="$(RISCV_SYSROOT)" \
		LD_LIBRARY_PATH="$(CURDIR)/candle-binding/target/$(RISCV_GNU_TARGET)/release" \
		$(RISCV_QEMU) -L $(RISCV_SYSROOT) "$(RISCV_QEMU_TEST)" \
		-test.run '$(BINDING_MINIMAL_GO_TESTS)' -test.skip '$(BINDING_MINIMAL_GO_SKIP)' -test.v=test2json -test.count=1 -test.timeout 90m \
		>"$(MODEL_TEST_REPORT_DIR)/binding.log" 2>&1 || status=$$?; \
		cat "$(MODEL_TEST_REPORT_DIR)/binding.log"; \
		go tool test2json -p candle-binding -t <"$(MODEL_TEST_REPORT_DIR)/binding.log" >"$(MODEL_TEST_REPORT_DIR)/qemu-minimal.jsonl" && \
		[ "$$status" -eq 0 ]
	@$(MAKE) build-router-riscv
	@echo "Starting the linux/riscv64 router under qemu-user"
	@RISCV_QEMU="$(RISCV_QEMU)" \
		RISCV_SYSROOT="$(RISCV_SYSROOT)" \
		RISCV_CANDLE_LIBDIR="$(RISCV_CANDLE_LIBDIR)" \
		RISCV_ROUTER_BIN="$(RISCV_ROUTER_BIN)" \
		RISCV_ROUTER_CONFIG="$(RISCV_ROUTER_CONFIG)" \
		RISCV_ROUTER_API_PORT="$(RISCV_ROUTER_API_PORT)" \
		MODEL_TEST_MANIFEST="$(RISCV_MODEL_MANIFEST)" \
		MODEL_TEST_REPORT_DIR="$(MODEL_TEST_REPORT_DIR)" \
		bash tools/ci/riscv-qemu-router-smoke.sh

# Candle-only linux/riscv64 router. Skips the ORT ABI check, does not
# link onnx/nlp/ml native libraries, and stubs valkey-glide (no libglide_ffi).
build-router-riscv: ## Cross-compile a Candle-only linux/riscv64 router
	@$(LOG_TARGET)
	@if ! command -v $(RISCV_GNU_CC) >/dev/null 2>&1; then \
		echo "missing $(RISCV_GNU_CC); install gcc-riscv64-linux-gnu"; \
		exit 1; \
	fi
	@if [ ! -f "$(RISCV_CANDLE_LIBDIR)/libcandle_semantic_router.so" ]; then \
		echo "Building Candle CPU library for $(RISCV_GNU_TARGET)"; \
		rustup target add $(RISCV_GNU_TARGET); \
		cd candle-binding && \
			CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=$(RISCV_GNU_CC) \
			CC_riscv64gc_unknown_linux_gnu=$(RISCV_GNU_CC) \
			CXX_riscv64gc_unknown_linux_gnu=$(RISCV_GNU_CXX) \
			cargo build --release --no-default-features --target $(RISCV_GNU_TARGET); \
	fi
	@bash tools/docker/check-native-abi.sh "$(RISCV_CANDLE_LIBDIR)/libcandle_semantic_router.so"
	@mkdir -p bin
	@echo "Building Candle-only router for linux/riscv64"
	@cd src/semantic-router && \
		CGO_ENABLED=1 GOOS=linux GOARCH=riscv64 CC=$(RISCV_GNU_CC) CXX=$(RISCV_GNU_CXX) \
		CGO_LDFLAGS= \
		go build -tags=milvus -o "$(RISCV_ROUTER_BIN)" ./cmd
