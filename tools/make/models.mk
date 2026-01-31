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

# mmBERT LoRA adapters (for Python fine-tuning)
MMBERT_LORA_ADAPTERS := \
	mmbert-intent-classifier-lora \
	mmbert-fact-check-lora \
	mmbert-pii-detector-lora \
	mmbert-jailbreak-detector-lora

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

download-mmbert-all: download-mmbert download-mmbert-lora download-mmbert-embedding download-mmbert-32k ## Download all mmBERT models, LoRA adapters, embedding, and 32K base model

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
	@for model in $(MMBERT_MODELS) $(MMBERT_LORA_ADAPTERS); do \
		rm -rf $(MODELS_DIR)/$$model; \
	done
	@rm -rf $(MODELS_DIR)/$(MMBERT_EMBEDDING_MODEL)
	@rm -rf $(MODELS_DIR)/$(MMBERT_32K_BASE_MODEL)
	@echo "✅ mmBERT models removed"
