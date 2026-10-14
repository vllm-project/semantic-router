# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

##@ Models

test-model-selection-parity: ## Compare Python-trained selectors with the router's selectors
	@python3 -m pytest -q src/training/model_selection/ml_model_selection/tests/test_selector_parity.py
	@python3 -m pytest -q src/training/model_selection/ml_model_selection/tests/test_service_boundary.py

.PHONY: test-model-selection-parity

ONNX_ARTIFACT_PYTHON_DEPS ?= onnx==1.22.0 onnxruntime==1.24.2

.PHONY: onnx-artifact-deps onnx-artifact-test
onnx-artifact-deps: harness-venv-install ## Install the ONNX artifact test dependencies into the harness venv
	@"$(AGENT_PYTHON)" -c "import onnx, onnxruntime" 2>/dev/null || "$(AGENT_PYTHON)" -m pip install --quiet $(ONNX_ARTIFACT_PYTHON_DEPS)

onnx-artifact-test: onnx-artifact-deps ## Verify external ONNX weight packing with real CPU inference
	@"$(AGENT_PYTHON)" -m unittest discover -s tools/models/onnx/artifact_tests -p 'test_*.py'

test-training-contracts: harness-venv-install ## Run dependency-light model training contract tests
	@"$(AGENT_PYTHON)" -m unittest src.training.control_plane.test_contracts
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s tools/models/onnx/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_embeddings/mmbert_32k/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_embeddings/multimodal/small/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_embeddings/multimodal/large/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/safety_classifier/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/user_feedback_classifier/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/escalation_risk/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/pii_model_fine_tuning_lora/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/sequence_repair/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_classifier/classifier_model_fine_tuning_lora/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/model_eval/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/training/kv_mapper/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m unittest discover -s src/kv_connector/tests -p 'test_*.py'
	@"$(AGENT_PYTHON)" -m pytest -q \
		bench/redteam \
		src/training/model_eval/test_provenance.py \
		src/training/model_eval/test_artifact_inventory.py \
		src/training/model_eval/test_baseline_artifact.py \
		src/training/model_eval/test_result_to_config.py \
		src/training/model_classifier/prompt_guard_fine_tuning_lora/test_jailbreak_provenance.py

# Models are downloaded by the model runtime the router starts. The targets
# below fetch training bases and adapters for the Python trainers.

# Hugging Face org for mmBERT models
HF_ORG := vllm-sr
MODELS_DIR := models

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

# The evaluation registry pins current Vela native snapshots.
.PHONY: download-eval-models
download-eval-models: ## Download Vela native eval models, including attack-only Guard (legacy is explicit)
	@python3 -m src.training.model_eval.download_models --output $(MODELS_DIR)

# The published-model contract and image calibration serve the runtime's pinned
# releases; the pinned Omni Nano snapshot is downloaded once into the models directory.
MODEL_TEST_MODELS_DIR ?= $(CURDIR)/$(MODELS_DIR)
MODEL_TEST_REPORT_DIR ?= $(CURDIR)/.agent-harness/model-tests/image-calibration
MODEL_TEST_MANIFEST ?= $(MODEL_TEST_REPORT_DIR)/models.json

download-models-image-calibration: ## Download the pinned Omni Nano release and attest its files
	@"$(AGENT_PYTHON)" tools/ci/prepare_model_test_assets.py \
		--variants nano --output "$(MODEL_TEST_MODELS_DIR)/vela-omni-artifacts"
	@"$(AGENT_PYTHON)" tools/ci/image_calibration.py --prepare-manifest \
		--artifact "$(MODEL_TEST_MODELS_DIR)/vela-omni-artifacts/vela-1.0-omni-nano" \
		--manifest "$(MODEL_TEST_MANIFEST)"

verify-image-routing-calibration: download-models-image-calibration ## Verify shipped image thresholds and the multimodal profile against source-bound fixtures
	@VLLM_SRUN_COMMAND="$${VLLM_SRUN_COMMAND:-$(AGENT_VENV)/bin/vllm-srun}" \
		"$(AGENT_PYTHON)" tools/ci/image_calibration.py \
		--manifest "$(MODEL_TEST_MANIFEST)" --output "$(MODEL_TEST_REPORT_DIR)"

.PHONY: download-models-image-calibration verify-image-routing-calibration

test-models: download-models-image-calibration ## Run the published-model contract through the model runtime
	@VLLM_SRUN_COMMAND="$${VLLM_SRUN_COMMAND:-$(AGENT_VENV)/bin/vllm-srun}" \
		"$(AGENT_PYTHON)" tools/ci/run_model_tests.py \
		--models-dir "$(MODEL_TEST_MODELS_DIR)" \
		--omni "$(MODEL_TEST_MODELS_DIR)/vela-omni-artifacts/vela-1.0-omni-nano" \
		--output "$(MODEL_TEST_REPORT_DIR)"

# After an intended change of the Router's Vela 2.0 questions, their fusion or
# a pinned revision: serve each pinned size on this CPU and rewrite the
# answers the contract compares with (needs model-runtime-install).
record-vela2-answers: ## Re-record the Vela 2.0 answers of the published-model contract
	@VLLM_SRUN_COMMAND="$${VLLM_SRUN_COMMAND:-$(AGENT_VENV)/bin/vllm-srun}" \
		"$(AGENT_PYTHON)" tools/ci/run_model_tests.py --record-vela2 \
		--models-dir "$(MODEL_TEST_MODELS_DIR)" \
		--omni "$(MODEL_TEST_MODELS_DIR)/vela-omni-artifacts/vela-1.0-omni-nano" \
		--output "$(MODEL_TEST_REPORT_DIR)"

.PHONY: test-models record-vela2-answers

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

download-mmbert-all: download-mmbert-lora download-mmbert-32k-lora download-mmbert-32k ## Download the mmBERT LoRA adapters and the 32K base model

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

clean-minimal-models: ## No-op target for backward compatibility
	@echo "ℹ️  This target is no longer needed"

clean-mmbert: ## Remove downloaded mmBERT models
	@echo "🗑️  Removing mmBERT models..."
	@for model in $(MMBERT_LORA_ADAPTERS) $(MMBERT_32K_LORA_ADAPTERS); do \
		rm -rf $(MODELS_DIR)/$$model; \
	done
	@rm -rf $(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL)
	@echo "mmBERT models removed"

# ======== mmBERT-32K Training ========
# Training targets for mmBERT-32K-YaRN fine-tuned models
# Base model: vllm-sr/mmbert-32k-yarn (32K context, multilingual)

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
# from vllm-sr/category-classifier-supplement
# Note: PII training uses AI4Privacy + Presidio combined dataset with char offset alignment

# Training script paths
TRAINING_DIR := src/training
LORA_DIR := $(TRAINING_DIR)/model_classifier

# Output directories for 32K models
MMBERT32K_MODELS_DIR := models/mmbert32k

train-mmbert32k-all: ## Train remaining legacy mmBERT-32K tasks (Guard retired)
	@echo "🚀 Training all mmBERT-32K models..."
	@echo "   Base model: vllm-sr/mmbert-32k-yarn"
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
		--model_name vllm-sr/mmbert-32k-yarn \
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

merge-mmbert32k-all: ## Merge all LoRA adapters into full models
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
				'vllm-sr/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-pii: ## Merge PII Detector LoRA adapter
	@echo "🔗 Merging PII Detector..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/pii-detector-lora" ]; then \
		python -c "from src.training.model_classifier.pii_model_fine_tuning_lora.pii_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/pii-detector-lora', \
				'$(MMBERT32K_MODELS_DIR)/pii-detector-merged', \
				'vllm-sr/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-jailbreak: ## Merge Jailbreak Detector LoRA adapter
	@echo "🔗 Merging Jailbreak Detector..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/jailbreak-detector-lora" ]; then \
		python -c "from src.training.model_classifier.prompt_guard_fine_tuning_lora.jailbreak_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/jailbreak-detector-lora', \
				'$(MMBERT32K_MODELS_DIR)/jailbreak-detector-merged', \
				'vllm-sr/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-factcheck: ## Merge Fact Check Classifier LoRA adapter
	@echo "🔗 Merging Fact Check Classifier..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/fact-check-lora" ]; then \
		python -c "from src.training.model_classifier.fact_check_fine_tuning_lora.fact_check_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/fact-check-lora', \
				'$(MMBERT32K_MODELS_DIR)/fact-check-merged', \
				'vllm-sr/mmbert-32k-yarn')"; \
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
