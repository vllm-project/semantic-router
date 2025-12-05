# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

##@ Models

# CI_MINIMAL_MODELS=true will download only the minimal set of models required for tests.
# Default behavior downloads the full set used for local development.

download-models:
download-models: ## Download models (full or minimal set depending on CI_MINIMAL_MODELS)
	@$(LOG_TARGET)
	@mkdir -p models
	@if [ "$$CI_MINIMAL_MODELS" = "true" ]; then \
		echo "CI_MINIMAL_MODELS=true -> downloading minimal model set"; \
		$(MAKE) -s download-models-minimal; \
	else \
		echo "CI_MINIMAL_MODELS not set -> downloading full model set"; \
		$(MAKE) -s download-models-full; \
	fi

# Minimal models needed to run unit tests on CI (avoid rate limits)
# - Category classifier (ModernBERT)
# - PII token classifier (ModernBERT Presidio)
# - Jailbreak classifier (ModernBERT)
# - Optional plain PII classifier mapping (small)
# - LoRA models (BERT architecture) for unified classifier tests

download-models-minimal:
download-models-minimal: ## Pre-download minimal set of models for CI tests
	@mkdir -p models
	# Pre-download tiny LLM for llm-katan (optional but speeds up first start)
	@if [ ! -f "models/Qwen/Qwen3-0.6B/.downloaded" ] || [ ! -d "models/Qwen/Qwen3-0.6B" ]; then \
		hf download Qwen/Qwen3-0.6B --local-dir models/Qwen/Qwen3-0.6B && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/Qwen/Qwen3-0.6B/.downloaded; \
	fi
	@if [ ! -f "models/all-MiniLM-L12-v2/.downloaded" ] || [ ! -d "models/all-MiniLM-L12-v2" ]; then \
		hf download sentence-transformers/all-MiniLM-L12-v2 --local-dir models/all-MiniLM-L12-v2 && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/all-MiniLM-L12-v2/.downloaded; \
	fi
	@if [ ! -f "models/category_classifier_modernbert-base_model/.downloaded" ] || [ ! -d "models/category_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/category_classifier_modernbert-base_model --local-dir models/category_classifier_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/category_classifier_modernbert-base_model/.downloaded; \
	fi
	@if [ ! -f "models/pii_classifier_modernbert-base_presidio_token_model/.downloaded" ] || [ ! -d "models/pii_classifier_modernbert-base_presidio_token_model" ]; then \
		hf download LLM-Semantic-Router/pii_classifier_modernbert-base_presidio_token_model --local-dir models/pii_classifier_modernbert-base_presidio_token_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/pii_classifier_modernbert-base_presidio_token_model/.downloaded; \
	fi
	@if [ ! -f "models/jailbreak_classifier_modernbert-base_model/.downloaded" ] || [ ! -d "models/jailbreak_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model --local-dir models/jailbreak_classifier_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/jailbreak_classifier_modernbert-base_model/.downloaded; \
	fi
	@if [ ! -f "models/pii_classifier_modernbert-base_model/.downloaded" ] || [ ! -d "models/pii_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/pii_classifier_modernbert-base_model --local-dir models/pii_classifier_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/pii_classifier_modernbert-base_model/.downloaded; \
	fi
	# Download LoRA models for unified classifier integration tests
	@if [ ! -f "models/lora_intent_classifier_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_intent_classifier_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model --local-dir models/lora_intent_classifier_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_intent_classifier_bert-base-uncased_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_pii_detector_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_pii_detector_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model --local-dir models/lora_pii_detector_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_pii_detector_bert-base-uncased_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_jailbreak_classifier_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_jailbreak_classifier_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model --local-dir models/lora_jailbreak_classifier_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_jailbreak_classifier_bert-base-uncased_model/.downloaded; \
	fi

# Full model set for local development and docs

download-models-full:
download-models-full: ## Download all models used in local development and docs
	@mkdir -p models
	# Pre-download tiny LLM for llm-katan (optional but speeds up first start)
	@if [ ! -f "models/Qwen/Qwen3-0.6B/.downloaded" ] || [ ! -d "models/Qwen/Qwen3-0.6B" ]; then \
		hf download Qwen/Qwen3-0.6B --local-dir models/Qwen/Qwen3-0.6B && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/Qwen/Qwen3-0.6B/.downloaded; \
	fi
	@if [ ! -f "models/all-MiniLM-L12-v2/.downloaded" ] || [ ! -d "models/all-MiniLM-L12-v2" ]; then \
		hf download sentence-transformers/all-MiniLM-L12-v2 --local-dir models/all-MiniLM-L12-v2 && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/all-MiniLM-L12-v2/.downloaded; \
	fi
	@if [ ! -f "models/category_classifier_modernbert-base_model/.downloaded" ] || [ ! -d "models/category_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/category_classifier_modernbert-base_model --local-dir models/category_classifier_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/category_classifier_modernbert-base_model/.downloaded; \
	fi
	@if [ ! -f "models/pii_classifier_modernbert-base_model/.downloaded" ] || [ ! -d "models/pii_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/pii_classifier_modernbert-base_model --local-dir models/pii_classifier_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/pii_classifier_modernbert-base_model/.downloaded; \
	fi
	@if [ ! -f "models/jailbreak_classifier_modernbert-base_model/.downloaded" ] || [ ! -d "models/jailbreak_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model --local-dir models/jailbreak_classifier_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/jailbreak_classifier_modernbert-base_model/.downloaded; \
	fi
	@if [ ! -f "models/pii_classifier_modernbert-base_presidio_token_model/.downloaded" ] || [ ! -d "models/pii_classifier_modernbert-base_presidio_token_model" ]; then \
		hf download LLM-Semantic-Router/pii_classifier_modernbert-base_presidio_token_model --local-dir models/pii_classifier_modernbert-base_presidio_token_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/pii_classifier_modernbert-base_presidio_token_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_intent_classifier_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_intent_classifier_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model --local-dir models/lora_intent_classifier_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_intent_classifier_bert-base-uncased_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_intent_classifier_roberta-base_model/.downloaded" ] || [ ! -d "models/lora_intent_classifier_roberta-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_intent_classifier_roberta-base_model --local-dir models/lora_intent_classifier_roberta-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_intent_classifier_roberta-base_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_intent_classifier_modernbert-base_model/.downloaded" ] || [ ! -d "models/lora_intent_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_intent_classifier_modernbert-base_model --local-dir models/lora_intent_classifier_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_intent_classifier_modernbert-base_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_pii_detector_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_pii_detector_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model --local-dir models/lora_pii_detector_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_pii_detector_bert-base-uncased_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_pii_detector_roberta-base_model/.downloaded" ] || [ ! -d "models/lora_pii_detector_roberta-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_pii_detector_roberta-base_model --local-dir models/lora_pii_detector_roberta-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_pii_detector_roberta-base_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_pii_detector_modernbert-base_model/.downloaded" ] || [ ! -d "models/lora_pii_detector_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_pii_detector_modernbert-base_model --local-dir models/lora_pii_detector_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_pii_detector_modernbert-base_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_jailbreak_classifier_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_jailbreak_classifier_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model --local-dir models/lora_jailbreak_classifier_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_jailbreak_classifier_bert-base-uncased_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_jailbreak_classifier_roberta-base_model/.downloaded" ] || [ ! -d "models/lora_jailbreak_classifier_roberta-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_jailbreak_classifier_roberta-base_model --local-dir models/lora_jailbreak_classifier_roberta-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_jailbreak_classifier_roberta-base_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_jailbreak_classifier_modernbert-base_model/.downloaded" ] || [ ! -d "models/lora_jailbreak_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_jailbreak_classifier_modernbert-base_model --local-dir models/lora_jailbreak_classifier_modernbert-base_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_jailbreak_classifier_modernbert-base_model/.downloaded; \
	fi
	@if [ ! -d "models/Qwen3-Embedding-0.6B" ]; then \
		hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B; \
	fi
	@if [ ! -d "models/embeddinggemma-300m" ]; then \
		echo "Attempting to download google/embeddinggemma-300m (may be restricted)..."; \
		hf download google/embeddinggemma-300m --local-dir models/embeddinggemma-300m || echo "⚠️  Warning: Failed to download embeddinggemma-300m (model may be restricted), continuing..."; \
	fi

# Download only LoRA and advanced embedding models (for CI after minimal tests)
download-models-lora:
download-models-lora: ## Download LoRA adapters and advanced embedding models only
	@mkdir -p models
	@echo "Downloading LoRA adapters and advanced embedding models..."
	@if [ ! -f "models/lora_intent_classifier_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_intent_classifier_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model --local-dir models/lora_intent_classifier_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_intent_classifier_bert-base-uncased_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_pii_detector_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_pii_detector_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model --local-dir models/lora_pii_detector_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_pii_detector_bert-base-uncased_model/.downloaded; \
	fi
	@if [ ! -f "models/lora_jailbreak_classifier_bert-base-uncased_model/.downloaded" ] || [ ! -d "models/lora_jailbreak_classifier_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model --local-dir models/lora_jailbreak_classifier_bert-base-uncased_model && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/lora_jailbreak_classifier_bert-base-uncased_model/.downloaded; \
	fi
	@if [ ! -d "models/Qwen3-Embedding-0.6B" ]; then \
		hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B; \
	fi
	@if [ ! -d "models/embeddinggemma-300m" ]; then \
		echo "Attempting to download google/embeddinggemma-300m (may be restricted)..."; \
		hf download google/embeddinggemma-300m --local-dir models/embeddinggemma-300m || echo "⚠️  Warning: Failed to download embeddinggemma-300m (model may be restricted), continuing..."; \
	fi

# Clean up minimal models to save disk space (for CI)
clean-minimal-models: ## Remove minimal models to save disk space
	@echo "Cleaning up minimal models to save disk space..."
	@rm -rf models/category_classifier_modernbert-base_model || true
	@rm -rf models/pii_classifier_modernbert-base_presidio_token_model || true
	@rm -rf models/jailbreak_classifier_modernbert-base_model || true
	@rm -rf models/pii_classifier_modernbert-base_model || true
	@echo "✓ Minimal models cleaned up"

# Convert models to OpenVINO format for testing
convert-openvino-test-models: ## Convert models to OpenVINO IR format for openvino-binding tests
	@echo "Converting models to OpenVINO IR format for tests..."
	@echo "================================================================"
	@echo "This will download HuggingFace models and convert to OpenVINO"
	@echo "================================================================"
	@mkdir -p openvino-binding/test_models
	
	# 1. Convert all-MiniLM-L6-v2 embedding model
	@echo "\n[1/2] Converting all-MiniLM-L6-v2 embedding model..."
	@if [ ! -f "openvino-binding/test_models/all-MiniLM-L6-v2/openvino_model.xml" ]; then \
		echo "  → Downloading HuggingFace model..."; \
		optimum-cli export openvino \
			--model sentence-transformers/all-MiniLM-L6-v2 \
			--task feature-extraction \
			openvino-binding/test_models/all-MiniLM-L6-v2 \
			--weight-format fp32 && \
		echo "  ✓ Converted: openvino-binding/test_models/all-MiniLM-L6-v2/openvino_model.xml"; \
	else \
		echo "  ✓ Already exists: openvino-binding/test_models/all-MiniLM-L6-v2/openvino_model.xml"; \
	fi
	
	# 2. Convert category_classifier_modernbert
	@echo "\n[2/2] Converting category_classifier_modernbert..."
	@if [ ! -f "openvino-binding/test_models/category_classifier_modernbert/openvino_model.xml" ]; then \
		echo "  → Downloading HuggingFace model..."; \
		optimum-cli export openvino \
			--model LLM-Semantic-Router/category_classifier_modernbert-base_model \
			--task text-classification \
			openvino-binding/test_models/category_classifier_modernbert \
			--weight-format fp32 && \
		echo "  ✓ Converted: openvino-binding/test_models/category_classifier_modernbert/openvino_model.xml"; \
	else \
		echo "  ✓ Already exists: openvino-binding/test_models/category_classifier_modernbert/openvino_model.xml"; \
	fi
	
	# 3. Convert tokenizers using openvino_tokenizers
	@echo "\n[3/3] Converting tokenizers to native OpenVINO format..."
	@if [ "$$SKIP_TOKENIZER_CONVERSION" = "1" ]; then \
		echo "  ⚠️  SKIP_TOKENIZER_CONVERSION=1 - skipping tokenizer conversion"; \
		echo "  Note: Tests will use fallback tokenization (slower but functional)"; \
	else \
		command -v python3 >/dev/null 2>&1 && PYTHON_CMD=python3 || PYTHON_CMD=python; \
		$$PYTHON_CMD openvino-binding/scripts/convert_test_tokenizers.py || { \
			echo ""; \
			echo "⚠️  Tokenizer conversion failed, but models are ready"; \
			echo "   Tests will use fallback tokenization"; \
			echo ""; \
			echo "To fix, install dependencies:"; \
			echo "  pip install openvino-tokenizers>=2025.3.0.0"; \
			echo ""; \
			echo "Or skip tokenizer conversion:"; \
			echo "  export SKIP_TOKENIZER_CONVERSION=1"; \
			echo "  make convert-openvino-test-models"; \
		}; \
	fi
	
	@echo "\n================================================================"
	@echo "✓ OpenVINO test models converted successfully!"
	@echo "================================================================"
	@echo "Models ready for testing:"
	@echo "  • openvino-binding/test_models/all-MiniLM-L6-v2/"
	@echo "  • openvino-binding/test_models/category_classifier_modernbert/"
	@echo ""
	@echo "Run tests with: cd openvino-binding && make test"
	@echo "================================================================"
