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

download-mmbert-all: download-mmbert download-mmbert-lora ## Download all mmBERT models and LoRA adapters

clean-minimal-models: ## No-op target for backward compatibility
	@echo "ℹ️  This target is no longer needed"

clean-mmbert: ## Remove downloaded mmBERT models
	@echo "🗑️  Removing mmBERT models..."
	@for model in $(MMBERT_MODELS) $(MMBERT_LORA_ADAPTERS); do \
		rm -rf $(MODELS_DIR)/$$model; \
	done
	@echo "✅ mmBERT models removed"
