# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

##@ Models

test-model-selection-parity: ## Compare Python-trained selectors with the current Rust C ABI
ifneq ($(PREBUILT_NATIVE_LIBS),1)
	@cargo test --locked --manifest-path ml-binding/Cargo.toml
endif
	@python3 -m pytest -q src/training/model_selection/ml_model_selection/tests/test_native_parity.py

.PHONY: test-model-selection-parity

.PHONY: onnx-artifact-test
onnx-artifact-test: ck-rewrite-deps ## Verify external ONNX weight packing with real CPU inference
	@"$(AGENT_PYTHON)" -m unittest discover -s onnx-binding/scripts/artifact_tests -p 'test_*.py'

test-training-contracts: harness-venv-install ## Run dependency-light model training contract tests
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s onnx-binding/scripts/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_embeddings/mmbert_32k/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_embeddings/multimodal/small/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_embeddings/multimodal/large/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/safety_classifier/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/user_feedback_classifier/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/pii_model_fine_tuning_lora/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/sequence_repair/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/classifier_model_fine_tuning_lora/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_eval/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/kv_mapper/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m pytest -q \
		src/training/model_eval/test_provenance.py \
		src/training/model_eval/test_artifact_inventory.py \
		src/training/model_eval/test_baseline_artifact.py \
		src/training/model_classifier/prompt_guard_fine_tuning_lora/test_jailbreak_provenance.py

# Models are automatically downloaded by the router at startup in production.
# For testing, we use the router's --download-only flag to download models and exit.

# Hugging Face org for mmBERT models
HF_ORG := llm-semantic-router
MODELS_DIR := models

# The checked-in reference suite targets
# peft-internal-testing/tiny-random-BertForSequenceClassification at
# 325bf1727142e5f4216ca8e3eef68752321979ac. The model remains external and
# must be downloaded at that exact revision before running qualification.
CANDLE_COMPAT_LABELS ?= LABEL_0,LABEL_1
CANDLE_COMPAT_SUITE ?= pkg/modelruntime/compatibility/testdata/tiny-random-bert-cpu-suite-v1.json
CANDLE_COMPAT_OUTPUT ?= $(CURDIR)/.agent-harness/compatibility/candle-cpu-receipt.json

# mmBERT merged models (for Rust inference)
MMBERT_MODELS := \
	mmbert-intent-classifier-merged \
	mmbert-fact-check-merged \
	mmbert-pii-detector-merged \
	mmbert-jailbreak-detector-merged

# mmBERT embedding model with 2D Matryoshka support
# Downloaded automatically by the router's built-in downloader (make download-models)
# Registry maps this to local path models/mom-embedding-ultra (see config/registry.go)
MMBERT_EMBEDDING_MODEL := mmbert-embed-32k-2d-matryoshka

# mmBERT base 32K YaRN model (extended context MLM model)
MMBERT_32K_BASE_MODEL := mmbert-32k-yarn

# mmBERT LoRA adapters (for Python fine-tuning) - 8K context
MMBERT_LORA_ADAPTERS := \
	mmbert-intent-classifier-lora \
	mmbert-fact-check-lora \
	mmbert-pii-detector-lora \
	mmbert-jailbreak-detector-lora

# mmBERT-32K LoRA adapters (32K context, YaRN-scaled)
MMBERT_32K_LORA_ADAPTERS := \
	mmbert32k-feedback-detector-lora \
	mmbert32k-intent-classifier-lora \
	mmbert32k-pii-detector-lora \
	mmbert32k-jailbreak-detector-lora \
	mmbert32k-factcheck-classifier-lora

# mmBERT-32K merged models (for Rust/Go inference)
MMBERT_32K_MERGED_MODELS := \
	mmbert32k-feedback-detector-merged \
	mmbert32k-intent-classifier-merged \
	mmbert32k-pii-detector-merged \
	mmbert32k-jailbreak-detector-merged \
	mmbert32k-factcheck-classifier-merged

# mmBERT-32K ONNX models (for ONNX Runtime inference - ROCm/CPU)
# These are stored in the onnx/ subdirectory of merged model repos
MMBERT_32K_ONNX_MODELS := \
	mmbert32k-intent-classifier-merged \
	mmbert32k-jailbreak-detector-merged \
	mmbert32k-pii-detector-merged \
	mmbert32k-factcheck-classifier-merged \
	mmbert32k-feedback-detector-merged

# Download models by running the router with --download-only flag
download-models: ## Download models using router's built-in download logic
	@echo "📦 Downloading models via router..."
	@echo ""
	@$(MAKE) build-router
	@echo ""
	@echo "Running router with --download-only flag..."
	@echo "This may take a few minutes depending on your network speed..."
	@export $(NATIVE_ENV) && \
		./bin/router -config=config/config.yaml --download-only
	@echo ""
	@echo "Models downloaded successfully"

QWEN3_EMBEDDING_REPO := Qwen/Qwen3-Embedding-0.6B
QWEN3_EMBEDDING_DIR := mom-embedding-pro

download-qwen3-embedding: ## Download the Qwen3 embedding model for binding tests and benchmarks
	@echo "⬇️  Downloading $(QWEN3_EMBEDDING_REPO)..."
	@mkdir -p "$(MODELS_DIR)"
	@hf download $(QWEN3_EMBEDDING_REPO) --local-dir "$(MODELS_DIR)/$(QWEN3_EMBEDDING_DIR)"

download-models-lora: ## Download models for LoRA and advanced embedding tests
	@$(MAKE) download-models
	@$(MAKE) download-qwen3-embedding

# The evaluation registry pins current Vela native snapshots. The MMBERT lists
# below and their download targets intentionally remain explicit legacy tools.
.PHONY: download-eval-models
download-eval-models: ## Download Vela native eval models, including attack-only Guard (legacy is explicit)
	@python3 -m src.training.model_eval.download_models --output $(MODELS_DIR)

.PHONY: qualify-candle-cpu check-candle-qualification-source test-modelcompat-native check-modelcompat

# Offline command tests are owned by the shared Go-tool registry. Native
# qualification stays explicit and never downloads a checkpoint by default.
test-modelcompat-native: ## Test the native receipt round-trip against an explicitly supplied pinned fixture
	@test -n "$(CANDLE_MODEL_PATH)" || (echo "CANDLE_MODEL_PATH is required" && exit 1)
	@$(MAKE) rust-ci
	@cd src/semantic-router && $(NATIVE_ENV) CGO_ENABLED=1 CANDLE_MODEL_PATH="$(abspath $(CANDLE_MODEL_PATH))" \
		go test -race -count=1 -v -run '^TestNativeCandleCPUCommandRoundTrip$$' ../../tools/modelcompat/*.go

check-modelcompat: test-modelcompat harness-go-bootstrap ## Test and lint the offline compatibility tool
	@cd src/semantic-router && $(NATIVE_ENV) "$$(go env GOPATH)/bin/golangci-lint" run --config ../../tools/linter/go/.golangci.yml ../../tools/modelcompat/*.go

# Do not attribute a working-tree build to HEAD. Local planning artifacts outside
# the compiled source trees do not affect this source check.
check-candle-qualification-source:
	@git diff --quiet HEAD -- || { echo "Candle qualification requires committed sources (tracked changes found)"; exit 1; }
	@untracked="$$(git ls-files --others --exclude-standard -- src/semantic-router candle-binding tools/modelcompat)" && \
		test -z "$$untracked" || { echo "Candle qualification requires committed sources (untracked source files found)"; exit 1; }

qualify-candle-cpu: check-candle-qualification-source ## Generate a local CPU Candle compatibility receipt (requires CANDLE_MODEL_PATH and CANDLE_ARTIFACT_REVISION)
	@test -n "$(CANDLE_MODEL_PATH)" || (echo "CANDLE_MODEL_PATH is required" && exit 1)
	@test -n "$(CANDLE_ARTIFACT_REVISION)" || (echo "CANDLE_ARTIFACT_REVISION is required" && exit 1)
	@$(MAKE) rust-ci
	@mkdir -p "$(dir $(CANDLE_COMPAT_OUTPUT))"
	@cd src/semantic-router && \
		$(NATIVE_ENV) \
		go run ../../tools/modelcompat/main.go qualify-candle-cpu \
			--model-path "$(abspath $(CANDLE_MODEL_PATH))" \
			--artifact-revision "$(CANDLE_ARTIFACT_REVISION)" \
			--router-revision "$(shell git rev-parse HEAD)" \
			--labels "$(CANDLE_COMPAT_LABELS)" \
			--suite "$(CANDLE_COMPAT_SUITE)" \
			--output "$(abspath $(CANDLE_COMPAT_OUTPUT))"
	@echo "Candle CPU compatibility receipt: $(CANDLE_COMPAT_OUTPUT)"

# Test artifacts use the runtime registry's pinned releases, independent of the
# example config's on-demand model graph. No Hugging Face IDs are duplicated here.
MODEL_TEST_PROVIDER ?= candle
MODEL_TEST_DEVICE ?= cpu
MODEL_TEST_REPORT_DIR ?= $(CURDIR)/.agent-harness/model-tests/$(MODEL_TEST_PROVIDER)-cpu
MODEL_TEST_MODELS_DIR ?= $(CURDIR)/$(MODELS_DIR)
MODEL_TEST_MANIFEST ?= $(MODEL_TEST_REPORT_DIR)/models.json
MULTIMODAL_TEST_REPORT_DIR ?= $(MODEL_TEST_REPORT_DIR)/multimodal
MULTIMODAL_TEST_MANIFEST ?= $(MULTIMODAL_TEST_REPORT_DIR)/models.json
PERF_MODEL_MANIFEST ?= $(CURDIR)/reports/models.json

download-models-test: ## Provision every supported runtime model at its registered revision
	@if [ "$(MODEL_TEST_PROVIDER)" = ort ]; then \
		CONTAINER_RUNTIME="$(CONTAINER_RUNTIME)" python3 tools/ci/prepare_model_test_assets.py \
			--variants nano mini --output "$(MODEL_TEST_MODELS_DIR)/vela-omni-artifacts"; \
	fi
	@cd src/semantic-router && go run ./tools/model-test-assets \
		--provider "$(MODEL_TEST_PROVIDER)" --suite runtime \
		--output "$(MODEL_TEST_MODELS_DIR)" --manifest "$(MODEL_TEST_MANIFEST)" --download

test-models: rust-ci download-models-test ## Require actual published-model CPU inference (MODEL_TEST_PROVIDER=candle|ort)
	@export $(NATIVE_ENV) && python3 tools/ci/run_model_tests.py \
		--manifest "$(MODEL_TEST_MANIFEST)" --output "$(MODEL_TEST_REPORT_DIR)" \
		--device "$(MODEL_TEST_DEVICE)"
	@if [ "$(MODEL_TEST_PROVIDER)" = candle ]; then \
		$(MAKE) test-multimodal-models; \
	fi

download-models-multimodal-test: ## Provision the pinned multimodal compatibility checkpoint
	@cd src/semantic-router && go run ./tools/model-test-assets \
		--provider candle --suite multimodal \
		--output "$(MODEL_TEST_MODELS_DIR)" --manifest "$(MULTIMODAL_TEST_MANIFEST)" --download

test-multimodal-models: rust-ci download-models-multimodal-test ## Require existing text/image compatibility contracts on Candle CPU
	@export $(NATIVE_ENV) && python3 tools/ci/run_model_tests.py --suite multimodal \
		--manifest "$(MULTIMODAL_TEST_MANIFEST)" --output "$(MULTIMODAL_TEST_REPORT_DIR)" \
		--device "$(MODEL_TEST_DEVICE)"

download-models-image-calibration: ## Prepare the pinned Nano ONNX artifact and attest its manifest
	@CONTAINER_RUNTIME="$(CONTAINER_RUNTIME)" python3 tools/ci/prepare_model_test_assets.py \
		--variants nano --output "$(MODEL_TEST_MODELS_DIR)/vela-omni-artifacts"
	@python3 tools/ci/image_calibration.py --prepare-manifest \
		--artifact "$(MODEL_TEST_MODELS_DIR)/vela-omni-artifacts/vela-1.0-omni-nano" \
		--manifest "$(MODEL_TEST_MANIFEST)"

verify-image-routing-calibration: rust-ci download-models-image-calibration ## Verify shipped image thresholds and the multimodal profile against source-bound fixtures
	@export $(NATIVE_ENV) && python3 tools/ci/image_calibration.py \
		--manifest "$(MODEL_TEST_MANIFEST)" --output "$(MODEL_TEST_REPORT_DIR)"

.PHONY: download-models-image-calibration verify-image-routing-calibration

download-models-perf: ## Provision the canonical Vela classifier and embedding benchmark artifacts
	@cd src/semantic-router && go run ./tools/model-test-assets \
		--provider candle --suite perf --output "$(CURDIR)/$(MODELS_DIR)" \
		--manifest "$(PERF_MODEL_MANIFEST)" --download

.PHONY: download-models-test test-models download-models-perf \
	download-models-multimodal-test test-multimodal-models

download-mmbert: ## Download all mmBERT merged models for Rust inference
	@echo "📦 Downloading mmBERT merged models from Hugging Face..."
	@mkdir -p $(MODELS_DIR)
	@for model in $(MMBERT_MODELS); do \
		echo ""; \
		echo "⬇️  Downloading $$model..."; \
		if [ -d "$(MODELS_DIR)/$$model" ]; then \
			echo "   Already exists, updating..."; \
		fi; \
		hf download $(HF_ORG)/$$model --local-dir $(MODELS_DIR)/$$model; \
	done
	@echo ""
	@echo "mmBERT models downloaded to $(MODELS_DIR)/"
	@ls -la $(MODELS_DIR)/

download-mmbert-lora: ## Download mmBERT LoRA adapters for Python fine-tuning
	@echo "📦 Downloading mmBERT LoRA adapters from Hugging Face..."
	@mkdir -p $(MODELS_DIR)
	@for adapter in $(MMBERT_LORA_ADAPTERS); do \
		echo ""; \
		echo "⬇️  Downloading $$adapter..."; \
		if [ -d "$(MODELS_DIR)/$$adapter" ]; then \
			echo "   Already exists, updating..."; \
		fi; \
		hf download $(HF_ORG)/$$adapter --local-dir $(MODELS_DIR)/$$adapter; \
	done
	@echo ""
	@echo "mmBERT LoRA adapters downloaded to $(MODELS_DIR)/"
	@ls -la $(MODELS_DIR)/

download-mmbert-all: download-mmbert download-mmbert-lora download-mmbert-32k-lora download-mmbert-32k-merged download-mmbert-32k download-mmbert-32k-onnx ## Download all mmBERT models, LoRA adapters, ONNX, and 32K base model

download-mmbert-32k-lora: ## Download mmBERT-32K LoRA adapters (32K context models)
	@echo "📦 Downloading mmBERT-32K LoRA adapters from Hugging Face..."
	@mkdir -p $(MODELS_DIR)
	@for adapter in $(MMBERT_32K_LORA_ADAPTERS); do \
		echo ""; \
		echo "⬇️  Downloading $$adapter..."; \
		if [ -d "$(MODELS_DIR)/$$adapter" ]; then \
			echo "   Already exists, updating..."; \
		fi; \
		hf download $(HF_ORG)/$$adapter --local-dir $(MODELS_DIR)/$$adapter; \
	done
	@echo ""
	@echo "mmBERT-32K LoRA adapters downloaded to $(MODELS_DIR)/"
	@echo ""
	@echo "Available 32K LoRA models:"
	@echo "  - mmbert32k-feedback-detector-lora   (4-class satisfaction)"
	@echo "  - mmbert32k-intent-classifier-lora   (MMLU-Pro categories)"
	@echo "  - mmbert32k-pii-detector-lora        (17 PII entity types)"
	@echo "  - mmbert32k-jailbreak-detector-lora  (prompt injection)"
	@echo "  - mmbert32k-factcheck-classifier-lora (fact-check routing)"

download-mmbert-32k-merged: ## Download mmBERT-32K merged models (for Rust/Go inference)
	@echo "📦 Downloading mmBERT-32K merged models from Hugging Face..."
	@echo "   These are full models for Rust/Go inference (not LoRA adapters)"
	@mkdir -p $(MODELS_DIR)
	@for model in $(MMBERT_32K_MERGED_MODELS); do \
		echo ""; \
		echo "⬇️  Downloading $$model..."; \
		if [ -d "$(MODELS_DIR)/$$model" ]; then \
			echo "   Already exists, updating..."; \
		fi; \
		hf download $(HF_ORG)/$$model --local-dir $(MODELS_DIR)/$$model; \
	done
	@echo ""
	@echo "mmBERT-32K merged models downloaded to $(MODELS_DIR)/"
	@echo ""
	@echo "Available 32K merged models (for Rust inference):"
	@echo "  - mmbert32k-feedback-detector-merged   (4-class satisfaction)"
	@echo "  - mmbert32k-intent-classifier-merged   (14-class MMLU-Pro)"
	@echo "  - mmbert32k-pii-detector-merged        (35-class PII NER)"
	@echo "  - mmbert32k-jailbreak-detector-merged  (binary jailbreak)"
	@echo "  - mmbert32k-factcheck-classifier-merged (binary fact-check)"

download-mmbert-32k: ## Download mmBERT 32K YaRN base model (extended context MLM)
	@echo "📦 Downloading mmBERT 32K YaRN base model..."
	@mkdir -p $(MODELS_DIR)
	@echo ""
	@echo "⬇️  Downloading $(MMBERT_32K_BASE_MODEL)..."
	@echo "   This model supports:"
	@echo "   - 32K context length (extended from 8K via YaRN RoPE scaling)"
	@echo "   - YaRN theta: 160000 (4x scaling from original)"
	@echo "   - Multilingual (1800+ languages via Glot500)"
	@echo "   - 307M parameters"
	@if [ -d "$(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL)" ]; then \
		echo "   Already exists, updating..."; \
	fi
	@hf download $(HF_ORG)/$(MMBERT_32K_BASE_MODEL) --local-dir $(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL)
	@echo ""
	@echo "mmBERT 32K YaRN model downloaded to $(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL)"
	@echo ""
	@echo "Model details:"
	@echo "  - Max context: 32,768 tokens"
	@echo "  - RoPE theta: 160,000 (YaRN-scaled)"
	@echo "  - Architecture: ModernBERT with Flash Attention 2"
	@echo "  - Reference: https://huggingface.co/$(HF_ORG)/$(MMBERT_32K_BASE_MODEL)"

download-mmbert-32k-onnx: ## Download mmBERT-32K ONNX models (from onnx/ subdir of merged repos on HF)
	@echo "📦 Downloading mmBERT-32K ONNX models from Hugging Face..."
	@echo "   These are ONNX versions for efficient inference with ONNX Runtime"
	@echo "   Supports: ROCm (AMD GPU), CUDA (NVIDIA GPU), OpenVINO (Intel), CPU"
	@mkdir -p $(MODELS_DIR)
	@for model in $(MMBERT_32K_ONNX_MODELS); do \
		echo ""; \
		echo "⬇️  Downloading $$model/onnx/..."; \
		onnx_dir="$(MODELS_DIR)/$${model}-onnx"; \
		if [ -d "$$onnx_dir" ]; then \
			echo "   Already exists, updating..."; \
		fi; \
		mkdir -p "$$onnx_dir"; \
		echo "   Downloading model.onnx and config.json from onnx/ subdir..."; \
		hf download $(HF_ORG)/$$model onnx/model.onnx onnx/config.json --local-dir "$$onnx_dir"; \
		if [ -f "$$onnx_dir/onnx/model.onnx" ]; then \
			mv "$$onnx_dir/onnx/"* "$$onnx_dir/"; \
			rmdir "$$onnx_dir/onnx" 2>/dev/null || true; \
		fi; \
		echo "   Downloading tokenizer files from onnx/ subdir (fallback to repo root)..."; \
		hf download $(HF_ORG)/$$model onnx/tokenizer.json onnx/tokenizer_config.json onnx/special_tokens_map.json --local-dir "$$onnx_dir" 2>/dev/null && { \
			for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do \
				[ -f "$$onnx_dir/onnx/$$f" ] && mv "$$onnx_dir/onnx/$$f" "$$onnx_dir/$$f"; \
			done; \
			rmdir "$$onnx_dir/onnx" 2>/dev/null || true; \
		} || { \
			echo "   (onnx/ subdir tokenizer not found, downloading from repo root)"; \
			hf download $(HF_ORG)/$$model tokenizer.json tokenizer_config.json special_tokens_map.json --local-dir "$$onnx_dir" 2>/dev/null || true; \
		}; \
		echo "   Downloading label/fact_check mapping files (if any)..."; \
		hf download $(HF_ORG)/$$model label_mapping.json --local-dir "$$onnx_dir" 2>/dev/null || true; \
		hf download $(HF_ORG)/$$model fact_check_mapping.json --local-dir "$$onnx_dir" 2>/dev/null || true; \
	done
	@echo ""
	@echo "mmBERT-32K ONNX models downloaded to $(MODELS_DIR)/"
	@echo ""
	@echo "Available ONNX models (~1.2GB each):"
	@echo "  - mmbert32k-intent-classifier-merged-onnx     (14-class MMLU-Pro)"
	@echo "  - mmbert32k-jailbreak-detector-merged-onnx    (binary jailbreak)"
	@echo "  - mmbert32k-pii-detector-merged-onnx          (35-class PII NER)"
	@echo "  - mmbert32k-factcheck-classifier-merged-onnx  (fact-check routing)"
	@echo "  - mmbert32k-feedback-detector-merged-onnx     (4-class satisfaction)"
	@echo ""
	@echo "Usage with onnx-binding:"
	@echo "  ONNX_MODEL_PATH=models/mmbert32k-intent-classifier-merged-onnx make run-router-onnx"

# Fact-check and feedback: download merged models, export to ONNX, upload to HF
# Requires: hf login (or HF_TOKEN) for upload
download-export-upload-onnx-factcheck-feedback: ## Download merged factcheck/feedback, export to ONNX, upload to Hugging Face
	@chmod +x scripts/download_export_upload_onnx_factcheck_feedback.sh
	@./scripts/download_export_upload_onnx_factcheck_feedback.sh

export-onnx-factcheck-feedback: ## Export factcheck and feedback merged models to ONNX (requires models in models/, run: make download-mmbert-32k-merged first; needs optimum: pip install optimum[onnxruntime])
	@echo "📤 Exporting factcheck and feedback to ONNX..."
	@python3 tools/models/export_classifiers_to_onnx.py --model factcheck --output-dir $(MODELS_DIR)
	@python3 tools/models/export_classifiers_to_onnx.py --model feedback --output-dir $(MODELS_DIR)
	@echo "ONNX export done: $(MODELS_DIR)/mmbert32k-factcheck-classifier-merged-onnx, $(MODELS_DIR)/mmbert32k-feedback-detector-merged-onnx"

upload-onnx-factcheck-feedback: export-onnx-factcheck-feedback ## Export and upload factcheck/feedback ONNX to Hugging Face (requires login)
	@echo "⬆️  Uploading ONNX to Hugging Face..."
	@SKIP_DOWNLOAD=1 SKIP_UPLOAD=0 ./scripts/download_export_upload_onnx_factcheck_feedback.sh

test-mmbert-32k: ## Test mmBERT 32K context with AVX512 optimization
	@echo "🧪 Testing mmBERT 32K context length support..."
	@echo "   Using release mode + native CPU optimization (AVX512)"
	@echo ""
	cd candle-binding && \
		MMBERT_MODEL_PATH=../$(MODELS_DIR)/mmbert-embed-32k-2d-matryoshka \
		RUSTFLAGS="-C target-cpu=native" \
		cargo test --release --no-default-features --lib test_32k_context_length -- --ignored --nocapture
	@echo ""
	@echo "mmBERT 32K context test completed"

test-mmbert-32k-all: ## Run all 32K-related tests with optimization
	@echo "🧪 Running all 32K tests with AVX512 optimization..."
	cd candle-binding && \
		MMBERT_MODEL_PATH=../$(MODELS_DIR)/mmbert-embed-32k-2d-matryoshka \
		RUSTFLAGS="-C target-cpu=native" \
		cargo test --release --no-default-features --lib "32k" -- --nocapture
	@echo ""
	@echo "All 32K tests completed"

clean-minimal-models: ## No-op target for backward compatibility
	@echo "ℹ️  This target is no longer needed"

clean-mmbert: ## Remove downloaded mmBERT models
	@echo "🗑️  Removing mmBERT models..."
	@for model in $(MMBERT_MODELS) $(MMBERT_LORA_ADAPTERS) $(MMBERT_32K_LORA_ADAPTERS) $(MMBERT_32K_MERGED_MODELS); do \
		rm -rf $(MODELS_DIR)/$$model; \
	done
	@for model in $(MMBERT_32K_ONNX_MODELS); do \
		rm -rf $(MODELS_DIR)/$${model}-onnx; \
	done
	@rm -rf $(MODELS_DIR)/$(MMBERT_EMBEDDING_MODEL)
	@rm -rf $(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL)
	@echo "mmBERT models removed"

# ======== mmBERT-32K Training ========
# Training targets for mmBERT-32K-YaRN fine-tuned models
# Base model: llm-semantic-router/mmbert-32k-yarn (32K context, multilingual)

##@ mmBERT-32K Training

# Training configuration (optimized for mmBERT-32K LoRA fine-tuning)
# Hyperparameters validated on 2026-02-02:
#   - Intent Classifier: 92% accuracy (MMLU-Pro + supplement data)
#   - PII Detector: 97.2% training accuracy (AI4Privacy + Presidio combined dataset)
#   - Feedback Detector: 98.8% accuracy (4-class, requires higher rank)
TRAIN_EPOCHS ?= 5
TRAIN_BATCH_SIZE ?= 16
TRAIN_LR ?= 2e-5
LORA_RANK ?= 32
LORA_ALPHA ?= 64
LORA_DROPOUT ?= 0.1
MAX_SAMPLES ?= 5000
WEIGHT_DECAY ?= 0.01

# Feedback Detector specific parameters (4-class requires higher capacity)
# Validated 2026-02-02: 98.83% accuracy, F1 macro 98.24%
# Higher rank needed to distinguish SAT/NEED_CLARIFICATION/WRONG_ANSWER/WANT_DIFFERENT
FEEDBACK_EPOCHS ?= 10
FEEDBACK_LR ?= 2e-5
FEEDBACK_LORA_RANK ?= 64
FEEDBACK_LORA_ALPHA ?= 128

# PII-specific training parameters (AI4Privacy + Presidio combined for best accuracy)
# AI4Privacy provides 400K diverse multilingual PII samples
# Combined with Presidio for entity type coverage
PII_EPOCHS ?= 8
PII_MAX_SAMPLES ?= 10000
PII_LR ?= 1e-4
PII_LORA_RANK ?= 48
PII_LORA_ALPHA ?= 96

# Note: Intent training includes supplement data (653 casual "other" samples)
# from LLM-Semantic-Router/category-classifier-supplement
# Note: PII training uses AI4Privacy + Presidio combined dataset with char offset alignment

# Training script paths
TRAINING_DIR := src/training
LORA_DIR := $(TRAINING_DIR)/model_classifier

# Output directories for 32K models
MMBERT32K_MODELS_DIR := models/mmbert32k

train-mmbert32k-all: ## Train remaining legacy mmBERT-32K tasks (Guard retired)
	@echo "🚀 Training all mmBERT-32K models..."
	@echo "   Base model: llm-semantic-router/mmbert-32k-yarn"
	@echo "   Epochs: $(TRAIN_EPOCHS), Batch size: $(TRAIN_BATCH_SIZE)"
	@echo ""
	@$(MAKE) train-mmbert32k-feedback
	@$(MAKE) train-mmbert32k-intent
	@$(MAKE) train-mmbert32k-pii
	@$(MAKE) train-mmbert32k-factcheck
	@echo ""
	@echo "All mmBERT-32K models trained successfully!"
	@echo ""
	@$(MAKE) list-mmbert32k-models

train-mmbert32k-feedback: ## Train Feedback Detector (4-class satisfaction)
	@echo "📊 Training Feedback Detector with mmBERT-32K..."
	@echo "   LoRA rank: $(FEEDBACK_LORA_RANK), alpha: $(FEEDBACK_LORA_ALPHA)"
	@echo "   Epochs: $(FEEDBACK_EPOCHS), LR: $(FEEDBACK_LR)"
	@echo "   (Higher rank needed for 4-class classification)"
	@mkdir -p models
	python $(TRAINING_DIR)/model_classifier/user_feedback_classifier/train_feedback_detector.py \
		--model_name llm-semantic-router/mmbert-32k-yarn \
		--output_dir models/mmbert32k-feedback-detector \
		--epochs $(FEEDBACK_EPOCHS) \
		--batch_size $(TRAIN_BATCH_SIZE) \
		--lr $(FEEDBACK_LR) \
		--use_lora \
		--lora_rank $(FEEDBACK_LORA_RANK) \
		--lora_alpha $(FEEDBACK_LORA_ALPHA) \
		--merge_lora
	@echo "Feedback Detector training complete (98.8% accuracy expected)"
	@echo "   LoRA: models/mmbert32k-feedback-detector-lora"
	@echo "   Merged: models/mmbert32k-feedback-detector-merged"

train-mmbert32k-intent: ## Train Intent Classifier (MMLU-Pro categories + supplement data)
	@echo "🎯 Training Intent Classifier with mmBERT-32K..."
	@echo "   LoRA rank: $(LORA_RANK), alpha: $(LORA_ALPHA)"
	@echo "   Includes supplement data for better 'other' category detection"
	@mkdir -p $(MMBERT32K_MODELS_DIR)
	python $(LORA_DIR)/classifier_model_fine_tuning_lora/ft_linear_lora.py \
		--mode train \
		--model mmbert-32k \
		--lora-rank $(LORA_RANK) \
		--lora-alpha $(LORA_ALPHA) \
		--epochs $(TRAIN_EPOCHS) \
		--batch-size $(TRAIN_BATCH_SIZE) \
		--learning-rate $(TRAIN_LR) \
		--max-samples $(MAX_SAMPLES)
	@echo "Intent Classifier training complete"
	@# Move to organized directory (handle both _model and non-_model suffixes)
	@if [ -d "lora_intent_classifier_mmbert-32k_r$(LORA_RANK)" ]; then \
		mv lora_intent_classifier_mmbert-32k_r$(LORA_RANK) $(MMBERT32K_MODELS_DIR)/intent-classifier-lora; \
	elif [ -d "lora_intent_classifier_mmbert-32k_r$(LORA_RANK)_model" ]; then \
		mv lora_intent_classifier_mmbert-32k_r$(LORA_RANK)_model $(MMBERT32K_MODELS_DIR)/intent-classifier-lora; \
	fi

train-mmbert32k-pii: ## Train PII Detector (AI4Privacy + Presidio combined dataset)
	@echo "Training PII Detector with mmBERT-32K (AI4Privacy + Presidio combined)..."
	@echo "   Dataset: AI4Privacy (70%) + Presidio (30%) for maximum coverage"
	@echo "   Epochs: $(PII_EPOCHS), Samples: $(PII_MAX_SAMPLES), LoRA rank: $(PII_LORA_RANK)"
	@mkdir -p $(MMBERT32K_MODELS_DIR)
	python $(LORA_DIR)/pii_model_fine_tuning_lora/pii_bert_finetuning_lora.py \
		--mode train \
		--model mmbert-32k \
		--lora-rank $(PII_LORA_RANK) \
		--lora-alpha $(PII_LORA_ALPHA) \
		--epochs $(PII_EPOCHS) \
		--batch-size $(TRAIN_BATCH_SIZE) \
		--learning-rate $(PII_LR) \
		--max-samples $(PII_MAX_SAMPLES) \
		--use-ai4privacy
	@echo "PII Detector training complete (97.2% accuracy expected)"
	@# Move to organized directory (handle both naming patterns)
	@if [ -d "lora_pii_detector_mmbert-32k_r$(PII_LORA_RANK)_token_model" ]; then \
		mv lora_pii_detector_mmbert-32k_r$(PII_LORA_RANK)_token_model $(MMBERT32K_MODELS_DIR)/pii-detector-lora; \
	elif [ -d "lora_pii_classifier_mmbert-32k_r$(PII_LORA_RANK)_model" ]; then \
		mv lora_pii_classifier_mmbert-32k_r$(PII_LORA_RANK)_model $(MMBERT32K_MODELS_DIR)/pii-detector-lora; \
	fi

train-mmbert32k-pii-quick: ## Quick PII training (3 epochs, 3000 samples)
	@echo "Quick PII Detector training (AI4Privacy + Presidio)..."
	python $(LORA_DIR)/pii_model_fine_tuning_lora/pii_bert_finetuning_lora.py \
		--mode train \
		--model mmbert-32k \
		--lora-rank 32 \
		--lora-alpha 64 \
		--epochs 3 \
		--batch-size 16 \
		--learning-rate 1e-4 \
		--max-samples 3000 \
		--use-ai4privacy
	@echo "Quick PII training complete"

train-mmbert32k-pii-presidio-only: ## Train PII Detector with Presidio only (legacy)
	@echo "Training PII Detector with Presidio only (legacy mode)..."
	python $(LORA_DIR)/pii_model_fine_tuning_lora/pii_bert_finetuning_lora.py \
		--mode train \
		--model mmbert-32k \
		--lora-rank $(LORA_RANK) \
		--lora-alpha $(LORA_ALPHA) \
		--epochs $(TRAIN_EPOCHS) \
		--batch-size $(TRAIN_BATCH_SIZE) \
		--learning-rate $(TRAIN_LR) \
		--max-samples $(MAX_SAMPLES) \
		--no-ai4privacy
	@echo "Presidio-only PII training complete"

train-mmbert32k-jailbreak: ## Retired: use the explicit Vela Guard sequence trainer
	@echo "Legacy Guard training is retired. Use the Vela Base with:"
	@echo "  python -m src.training.model_classifier.sequence_repair.train --method full --fresh-head"
	@echo "Supply --base, --base-id, --base-revision, --contract, --train, --dev, and --output explicitly."
	@echo "See src/training/model_classifier/prompt_guard_fine_tuning_lora/README.md."
	@exit 2

train-mmbert32k-factcheck: ## Train Fact Check Classifier
	@echo "Training Fact Check Classifier with mmBERT-32K..."
	@mkdir -p $(MMBERT32K_MODELS_DIR)
	python $(LORA_DIR)/fact_check_fine_tuning_lora/fact_check_bert_finetuning_lora.py \
		--mode train \
		--model mmbert-32k \
		--lora-rank $(LORA_RANK) \
		--lora-alpha $(LORA_ALPHA) \
		--epochs $(TRAIN_EPOCHS) \
		--batch-size $(TRAIN_BATCH_SIZE) \
		--learning-rate $(TRAIN_LR) \
		--max-samples $(MAX_SAMPLES)
	@echo "Fact Check Classifier training complete"
	@# Move to organized directory
	@if [ -d "lora_fact_check_classifier_mmbert-32k_r$(LORA_RANK)_model" ]; then \
		mv lora_fact_check_classifier_mmbert-32k_r$(LORA_RANK)_model $(MMBERT32K_MODELS_DIR)/fact-check-lora; \
	fi

merge-mmbert32k-all: ## Merge all LoRA adapters into full models for Rust inference
	@echo "🔗 Merging all mmBERT-32K LoRA adapters..."
	@echo ""
	@$(MAKE) merge-mmbert32k-intent
	@$(MAKE) merge-mmbert32k-pii
	@$(MAKE) merge-mmbert32k-jailbreak
	@$(MAKE) merge-mmbert32k-factcheck
	@echo ""
	@echo "All LoRA adapters merged!"
	@$(MAKE) list-mmbert32k-models

merge-mmbert32k-intent: ## Merge Intent Classifier LoRA adapter
	@echo "🔗 Merging Intent Classifier..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/intent-classifier-lora" ]; then \
		python -c "from src.training.model_classifier.classifier_model_fine_tuning_lora.ft_linear_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/intent-classifier-lora', \
				'$(MMBERT32K_MODELS_DIR)/intent-classifier-merged', \
				'llm-semantic-router/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-pii: ## Merge PII Detector LoRA adapter
	@echo "🔗 Merging PII Detector..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/pii-detector-lora" ]; then \
		python -c "from src.training.model_classifier.pii_model_fine_tuning_lora.pii_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/pii-detector-lora', \
				'$(MMBERT32K_MODELS_DIR)/pii-detector-merged', \
				'llm-semantic-router/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-jailbreak: ## Merge Jailbreak Detector LoRA adapter
	@echo "🔗 Merging Jailbreak Detector..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/jailbreak-detector-lora" ]; then \
		python -c "from src.training.model_classifier.prompt_guard_fine_tuning_lora.jailbreak_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/jailbreak-detector-lora', \
				'$(MMBERT32K_MODELS_DIR)/jailbreak-detector-merged', \
				'llm-semantic-router/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-factcheck: ## Merge Fact Check Classifier LoRA adapter
	@echo "🔗 Merging Fact Check Classifier..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/fact-check-lora" ]; then \
		python -c "from src.training.model_classifier.fact_check_fine_tuning_lora.fact_check_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/fact-check-lora', \
				'$(MMBERT32K_MODELS_DIR)/fact-check-merged', \
				'llm-semantic-router/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

list-mmbert32k-models: ## List all trained mmBERT-32K models
	@echo ""
	@echo "📦 Trained mmBERT-32K Models:"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if [ -d "$(MMBERT32K_MODELS_DIR)" ]; then \
		ls -la $(MMBERT32K_MODELS_DIR)/ 2>/dev/null || echo "   (empty)"; \
	else \
		echo "   No models trained yet. Run: make train-mmbert32k-all"; \
	fi
	@echo ""

clean-mmbert32k: ## Remove all trained mmBERT-32K models
	@echo "🗑️  Removing trained mmBERT-32K models..."
	@rm -rf $(MMBERT32K_MODELS_DIR)
	@rm -rf lora_*_mmbert-32k_*
	@echo "mmBERT-32K models removed"

##@ mmBERT-32K GPU Training (ROCm)

# Docker image for GPU training
ROCM_IMAGE ?= rocm/vllm:v0.14.0_amd_dev

train-mmbert32k-gpu: ## Train all mmBERT-32K models on GPU (ROCm Docker)
	@echo "🚀 Training mmBERT-32K models on GPU..."
	@./src/training/model_classifier/train-mmbert32k-gpu.sh

train-mmbert32k-gpu-quick: ## Quick GPU training (fewer samples, 3 epochs)
	@echo "🚀 Quick GPU training (3 epochs, 2000 samples)..."
	TRAIN_EPOCHS=3 MAX_SAMPLES=2000 ./src/training/model_classifier/train-mmbert32k-gpu.sh

train-mmbert32k-gpu-full: ## Full GPU training (more samples, 10 epochs)
	@echo "🚀 Full GPU training (10 epochs, 20000 samples)..."
	TRAIN_EPOCHS=10 MAX_SAMPLES=20000 TRAIN_BATCH_SIZE=32 ./src/training/model_classifier/train-mmbert32k-gpu.sh

train-mmbert32k-gpu-shell: ## Open interactive shell in GPU training container
	@echo "🐚 Opening interactive shell in ROCm container..."
	@docker run --rm -it \
		--device=/dev/kfd \
		--device=/dev/dri \
		--group-add video \
		--shm-size=16g \
		-v "$(CURDIR):/workspace" \
		-v "$(HOME)/.cache/huggingface:/root/.cache/huggingface" \
		-w /workspace \
		-e HF_HOME="/root/.cache/huggingface" \
		$(ROCM_IMAGE) \
		/bin/bash

check-gpu: ## Check GPU availability in Docker container
	@echo "🔍 Checking GPU availability..."
	@docker run --rm \
		--device=/dev/kfd \
		--device=/dev/dri \
		--group-add video \
		$(ROCM_IMAGE) \
		python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1024**3:.0f}GB)') for i in range(torch.cuda.device_count())]"

# Convert only the immutable registered Vela graphs; tokenizer failure is fatal.
convert-openvino-test-models: ## Convert pinned Vela Domain and Embedding ONNX graphs to OpenVINO IR
	@mkdir -p "$(OPENVINO_TEST_REPORT_DIR)"
	@rm -f "$(OPENVINO_TEST_REPORT_DIR)/models.json" "$(OPENVINO_TEST_REPORT_DIR)/inference.json"
	@$(OPENVINO_PYTHON) -m unittest discover -s openvino-binding/scripts -p convert_published_models_test.py
	@cd src/semantic-router && go run ./tools/model-test-assets \
		--suite openvino --provider ort --output "$(OPENVINO_TEST_SOURCE_DIR)" \
		--manifest "$(OPENVINO_TEST_REPORT_DIR)/sources.json" --download
	@$(OPENVINO_PYTHON) openvino-binding/scripts/convert_published_models.py \
		--sources "$(OPENVINO_TEST_REPORT_DIR)/sources.json" \
		--output "$(OPENVINO_TEST_MODEL_DIR)" \
		--manifest "$(OPENVINO_TEST_REPORT_DIR)/models.json"
