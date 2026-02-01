# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

##@ Models

# Models are automatically downloaded by the router at startup in production.
# For testing, we use the router's --download-only flag to download models and exit.

# Hugging Face org for mmBERT models
HF_ORG := llm-semantic-router
MODELS_DIR := models

# mmBERT merged models (for Rust inference)
MMBERT_MODELS := \
	mmbert-intent-classifier-merged \
	mmbert-fact-check-merged \
	mmbert-pii-detector-merged \
	mmbert-jailbreak-detector-merged

# mmBERT embedding model with 2D Matryoshka support
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

# Download models by running the router with --download-only flag
download-models: ## Download models using router's built-in download logic
	@echo "📦 Downloading models via router..."
	@echo ""
	@$(MAKE) build-router
	@echo ""
	@echo "Running router with --download-only flag..."
	@echo "This may take a few minutes depending on your network speed..."
	@./bin/router -config=config/config.yaml --download-only
	@echo ""
	@echo "✅ Models downloaded successfully"

download-models-lora: ## Download LoRA models (same as download-models now)
	@$(MAKE) download-models

download-mmbert: ## Download all mmBERT merged models for Rust inference
	@echo "📦 Downloading mmBERT merged models from Hugging Face..."
	@mkdir -p $(MODELS_DIR)
	@for model in $(MMBERT_MODELS); do \
		echo ""; \
		echo "⬇️  Downloading $$model..."; \
		if [ -d "$(MODELS_DIR)/$$model" ]; then \
			echo "   Already exists, updating..."; \
		fi; \
		huggingface-cli download $(HF_ORG)/$$model --local-dir $(MODELS_DIR)/$$model --local-dir-use-symlinks False; \
	done
	@echo ""
	@echo "✅ mmBERT models downloaded to $(MODELS_DIR)/"
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
		huggingface-cli download $(HF_ORG)/$$adapter --local-dir $(MODELS_DIR)/$$adapter --local-dir-use-symlinks False; \
	done
	@echo ""
	@echo "✅ mmBERT LoRA adapters downloaded to $(MODELS_DIR)/"
	@ls -la $(MODELS_DIR)/

download-mmbert-all: download-mmbert download-mmbert-lora download-mmbert-32k-lora download-mmbert-32k-merged download-mmbert-embedding download-mmbert-32k ## Download all mmBERT models, LoRA adapters, embedding, and 32K base model

download-mmbert-32k-lora: ## Download mmBERT-32K LoRA adapters (32K context models)
	@echo "📦 Downloading mmBERT-32K LoRA adapters from Hugging Face..."
	@mkdir -p $(MODELS_DIR)
	@for adapter in $(MMBERT_32K_LORA_ADAPTERS); do \
		echo ""; \
		echo "⬇️  Downloading $$adapter..."; \
		if [ -d "$(MODELS_DIR)/$$adapter" ]; then \
			echo "   Already exists, updating..."; \
		fi; \
		huggingface-cli download $(HF_ORG)/$$adapter --local-dir $(MODELS_DIR)/$$adapter --local-dir-use-symlinks False; \
	done
	@echo ""
	@echo "✅ mmBERT-32K LoRA adapters downloaded to $(MODELS_DIR)/"
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
		huggingface-cli download $(HF_ORG)/$$model --local-dir $(MODELS_DIR)/$$model --local-dir-use-symlinks False; \
	done
	@echo ""
	@echo "✅ mmBERT-32K merged models downloaded to $(MODELS_DIR)/"
	@echo ""
	@echo "Available 32K merged models (for Rust inference):"
	@echo "  - mmbert32k-feedback-detector-merged   (4-class satisfaction)"
	@echo "  - mmbert32k-intent-classifier-merged   (14-class MMLU-Pro)"
	@echo "  - mmbert32k-pii-detector-merged        (35-class PII NER)"
	@echo "  - mmbert32k-jailbreak-detector-merged  (binary jailbreak)"
	@echo "  - mmbert32k-factcheck-classifier-merged (binary fact-check)"

download-mmbert-embedding: ## Download mmBERT 2D Matryoshka embedding model
	@echo "📦 Downloading mmBERT 2D Matryoshka embedding model..."
	@mkdir -p $(MODELS_DIR)
	@echo ""
	@echo "⬇️  Downloading $(MMBERT_EMBEDDING_MODEL)..."
	@echo "   This model supports:"
	@echo "   - 32K context length (YaRN-scaled RoPE)"
	@echo "   - Multilingual (1800+ languages)"
	@echo "   - 2D Matryoshka: layer early exit (3/6/11/22) + dimension reduction (64-768)"
	@if [ -d "$(MODELS_DIR)/$(MMBERT_EMBEDDING_MODEL)" ]; then \
		echo "   Already exists, updating..."; \
	fi
	@huggingface-cli download $(HF_ORG)/$(MMBERT_EMBEDDING_MODEL) --local-dir $(MODELS_DIR)/$(MMBERT_EMBEDDING_MODEL) --local-dir-use-symlinks False
	@echo ""
	@echo "✅ mmBERT embedding model downloaded to $(MODELS_DIR)/$(MMBERT_EMBEDDING_MODEL)"
	@echo ""
	@echo "Usage example:"
	@echo "  make run-router CONFIG_FILE=config/intelligent-routing/in-tree/embedding-mmbert.yaml"

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
	@huggingface-cli download $(HF_ORG)/$(MMBERT_32K_BASE_MODEL) --local-dir $(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL) --local-dir-use-symlinks False
	@echo ""
	@echo "✅ mmBERT 32K YaRN model downloaded to $(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL)"
	@echo ""
	@echo "Model details:"
	@echo "  - Max context: 32,768 tokens"
	@echo "  - RoPE theta: 160,000 (YaRN-scaled)"
	@echo "  - Architecture: ModernBERT with Flash Attention 2"
	@echo "  - Reference: https://huggingface.co/$(HF_ORG)/$(MMBERT_32K_BASE_MODEL)"

test-mmbert-32k: ## Test mmBERT 32K context with AVX512 optimization
	@echo "🧪 Testing mmBERT 32K context length support..."
	@echo "   Using release mode + native CPU optimization (AVX512)"
	@echo ""
	cd candle-binding && \
		MMBERT_MODEL_PATH=../$(MODELS_DIR)/mmbert-embed-32k-2d-matryoshka \
		RUSTFLAGS="-C target-cpu=native" \
		cargo test --release --no-default-features --lib test_32k_context_length -- --ignored --nocapture
	@echo ""
	@echo "✅ mmBERT 32K context test completed"

test-mmbert-32k-all: ## Run all 32K-related tests with optimization
	@echo "🧪 Running all 32K tests with AVX512 optimization..."
	cd candle-binding && \
		MMBERT_MODEL_PATH=../$(MODELS_DIR)/mmbert-embed-32k-2d-matryoshka \
		RUSTFLAGS="-C target-cpu=native" \
		cargo test --release --no-default-features --lib "32k" -- --nocapture
	@echo ""
	@echo "✅ All 32K tests completed"

clean-minimal-models: ## No-op target for backward compatibility
	@echo "ℹ️  This target is no longer needed"

clean-mmbert: ## Remove downloaded mmBERT models
	@echo "🗑️  Removing mmBERT models..."
	@for model in $(MMBERT_MODELS) $(MMBERT_LORA_ADAPTERS) $(MMBERT_32K_LORA_ADAPTERS) $(MMBERT_32K_MERGED_MODELS); do \
		rm -rf $(MODELS_DIR)/$$model; \
	done
	@rm -rf $(MODELS_DIR)/$(MMBERT_EMBEDDING_MODEL)
	@rm -rf $(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL)
	@echo "✅ mmBERT models removed"

# ======== mmBERT-32K Training ========
# Training targets for mmBERT-32K-YaRN fine-tuned models
# Base model: llm-semantic-router/mmbert-32k-yarn (32K context, multilingual)

##@ mmBERT-32K Training

# Training configuration (optimized for mmBERT-32K LoRA fine-tuning)
# Hyperparameters validated on 2026-02-01:
#   - Intent Classifier: 92% accuracy (MMLU-Pro + supplement data)
#   - Jailbreak Detector: 97.7% training accuracy (toxic-chat + salad-data)
#   - PII Detector: 97.2% training accuracy (AI4Privacy + Presidio combined dataset)
TRAIN_EPOCHS ?= 5
TRAIN_BATCH_SIZE ?= 16
TRAIN_LR ?= 2e-4
LORA_RANK ?= 32
LORA_ALPHA ?= 64
LORA_DROPOUT ?= 0.1
MAX_SAMPLES ?= 5000
WEIGHT_DECAY ?= 0.1

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
LORA_DIR := $(TRAINING_DIR)/training_lora

# Output directories for 32K models
MMBERT32K_MODELS_DIR := models/mmbert32k

train-mmbert32k-all: ## Train all mmBERT-32K models (LoRA + Merged)
	@echo "🚀 Training all mmBERT-32K models..."
	@echo "   Base model: llm-semantic-router/mmbert-32k-yarn"
	@echo "   Epochs: $(TRAIN_EPOCHS), Batch size: $(TRAIN_BATCH_SIZE)"
	@echo ""
	@$(MAKE) train-mmbert32k-feedback
	@$(MAKE) train-mmbert32k-intent
	@$(MAKE) train-mmbert32k-pii
	@$(MAKE) train-mmbert32k-jailbreak
	@$(MAKE) train-mmbert32k-factcheck
	@echo ""
	@echo "✅ All mmBERT-32K models trained successfully!"
	@echo ""
	@$(MAKE) list-mmbert32k-models

train-mmbert32k-feedback: ## Train Feedback Detector (4-class satisfaction)
	@echo "📊 Training Feedback Detector with mmBERT-32K..."
	@mkdir -p $(MMBERT32K_MODELS_DIR)
	python $(TRAINING_DIR)/modernbert_dissat_pipeline/train_feedback_detector.py \
		--model_name llm-semantic-router/mmbert-32k-yarn \
		--output_dir $(MMBERT32K_MODELS_DIR)/feedback-detector \
		--epochs $(TRAIN_EPOCHS) \
		--batch_size $(TRAIN_BATCH_SIZE) \
		--lr $(TRAIN_LR) \
		--use_lora \
		--lora_rank $(LORA_RANK) \
		--lora_alpha $(LORA_ALPHA) \
		--merge_lora
	@echo "✅ Feedback Detector training complete"
	@echo "   LoRA: $(MMBERT32K_MODELS_DIR)/feedback-detector_lora"
	@echo "   Merged: $(MMBERT32K_MODELS_DIR)/feedback-detector_merged"

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
	@echo "✅ Intent Classifier training complete"
	@# Move to organized directory (handle both _model and non-_model suffixes)
	@if [ -d "lora_intent_classifier_mmbert-32k_r$(LORA_RANK)" ]; then \
		mv lora_intent_classifier_mmbert-32k_r$(LORA_RANK) $(MMBERT32K_MODELS_DIR)/intent-classifier-lora; \
	elif [ -d "lora_intent_classifier_mmbert-32k_r$(LORA_RANK)_model" ]; then \
		mv lora_intent_classifier_mmbert-32k_r$(LORA_RANK)_model $(MMBERT32K_MODELS_DIR)/intent-classifier-lora; \
	fi

train-mmbert32k-pii: ## Train PII Detector (AI4Privacy + Presidio combined dataset)
	@echo "🔒 Training PII Detector with mmBERT-32K (AI4Privacy + Presidio combined)..."
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
	@echo "✅ PII Detector training complete (97.2% accuracy expected)"
	@# Move to organized directory (handle both naming patterns)
	@if [ -d "lora_pii_detector_mmbert-32k_r$(PII_LORA_RANK)_token_model" ]; then \
		mv lora_pii_detector_mmbert-32k_r$(PII_LORA_RANK)_token_model $(MMBERT32K_MODELS_DIR)/pii-detector-lora; \
	elif [ -d "lora_pii_classifier_mmbert-32k_r$(PII_LORA_RANK)_model" ]; then \
		mv lora_pii_classifier_mmbert-32k_r$(PII_LORA_RANK)_model $(MMBERT32K_MODELS_DIR)/pii-detector-lora; \
	fi

train-mmbert32k-pii-quick: ## Quick PII training (3 epochs, 3000 samples)
	@echo "🔒 Quick PII Detector training (AI4Privacy + Presidio)..."
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
	@echo "✅ Quick PII training complete"

train-mmbert32k-pii-presidio-only: ## Train PII Detector with Presidio only (legacy)
	@echo "🔒 Training PII Detector with Presidio only (legacy mode)..."
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
	@echo "✅ Presidio-only PII training complete"

train-mmbert32k-jailbreak: ## Train Jailbreak Detector (toxic-chat + salad-data)
	@echo "🛡️  Training Jailbreak Detector with mmBERT-32K..."
	@mkdir -p $(MMBERT32K_MODELS_DIR)
	python $(LORA_DIR)/prompt_guard_fine_tuning_lora/jailbreak_bert_finetuning_lora.py \
		--mode train \
		--model mmbert-32k \
		--lora-rank $(LORA_RANK) \
		--lora-alpha $(LORA_ALPHA) \
		--epochs $(TRAIN_EPOCHS) \
		--batch-size $(TRAIN_BATCH_SIZE) \
		--learning-rate $(TRAIN_LR) \
		--max-samples $(MAX_SAMPLES)
	@echo "✅ Jailbreak Detector training complete (97.7% accuracy expected)"
	@# Move to organized directory
	@if [ -d "lora_jailbreak_classifier_mmbert-32k_r$(LORA_RANK)_model" ]; then \
		mv lora_jailbreak_classifier_mmbert-32k_r$(LORA_RANK)_model $(MMBERT32K_MODELS_DIR)/jailbreak-detector-lora; \
	fi

train-mmbert32k-factcheck: ## Train Fact Check Classifier
	@echo "✓ Training Fact Check Classifier with mmBERT-32K..."
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
	@echo "✅ Fact Check Classifier training complete"
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
	@echo "✅ All LoRA adapters merged!"
	@$(MAKE) list-mmbert32k-models

merge-mmbert32k-intent: ## Merge Intent Classifier LoRA adapter
	@echo "🔗 Merging Intent Classifier..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/intent-classifier-lora" ]; then \
		python -c "from src.training.training_lora.classifier_model_fine_tuning_lora.ft_linear_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/intent-classifier-lora', \
				'$(MMBERT32K_MODELS_DIR)/intent-classifier-merged', \
				'llm-semantic-router/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-pii: ## Merge PII Detector LoRA adapter
	@echo "🔗 Merging PII Detector..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/pii-detector-lora" ]; then \
		python -c "from src.training.training_lora.pii_model_fine_tuning_lora.pii_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/pii-detector-lora', \
				'$(MMBERT32K_MODELS_DIR)/pii-detector-merged', \
				'llm-semantic-router/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-jailbreak: ## Merge Jailbreak Detector LoRA adapter
	@echo "🔗 Merging Jailbreak Detector..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/jailbreak-detector-lora" ]; then \
		python -c "from src.training.training_lora.prompt_guard_fine_tuning_lora.jailbreak_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
			merge_lora_adapter_to_full_model('$(MMBERT32K_MODELS_DIR)/jailbreak-detector-lora', \
				'$(MMBERT32K_MODELS_DIR)/jailbreak-detector-merged', \
				'llm-semantic-router/mmbert-32k-yarn')"; \
	else \
		echo "   ⚠️  LoRA adapter not found, skipping..."; \
	fi

merge-mmbert32k-factcheck: ## Merge Fact Check Classifier LoRA adapter
	@echo "🔗 Merging Fact Check Classifier..."
	@if [ -d "$(MMBERT32K_MODELS_DIR)/fact-check-lora" ]; then \
		python -c "from src.training.training_lora.fact_check_fine_tuning_lora.fact_check_bert_finetuning_lora import merge_lora_adapter_to_full_model; \
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
	@echo "✅ mmBERT-32K models removed"

##@ mmBERT-32K GPU Training (ROCm)

# Docker image for GPU training
ROCM_IMAGE ?= rocm/vllm:v0.14.0_amd_dev

train-mmbert32k-gpu: ## Train all mmBERT-32K models on GPU (ROCm Docker)
	@echo "🚀 Training mmBERT-32K models on GPU..."
	@./scripts/train-mmbert32k-gpu.sh

train-mmbert32k-gpu-quick: ## Quick GPU training (fewer samples, 3 epochs)
	@echo "🚀 Quick GPU training (3 epochs, 2000 samples)..."
	TRAIN_EPOCHS=3 MAX_SAMPLES=2000 ./scripts/train-mmbert32k-gpu.sh

train-mmbert32k-gpu-full: ## Full GPU training (more samples, 10 epochs)
	@echo "🚀 Full GPU training (10 epochs, 20000 samples)..."
	TRAIN_EPOCHS=10 MAX_SAMPLES=20000 TRAIN_BATCH_SIZE=32 ./scripts/train-mmbert32k-gpu.sh

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
