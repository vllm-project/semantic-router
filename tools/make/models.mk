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
# - Embedding models (Qwen3-Embedding-0.6B) for smart embedding tests
# Note: embeddinggemma-300m is gated and requires HF_TOKEN, so it's excluded from CI

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
	# Download embedding models for smart embedding tests (Qwen3 only - Gemma is gated)
	@if [ ! -f "models/Qwen3-Embedding-0.6B/.downloaded" ] || [ ! -d "models/Qwen3-Embedding-0.6B" ]; then \
		hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/Qwen3-Embedding-0.6B/.downloaded; \
	fi
	# Download hallucination mitigation models
	# Hallucination detection model (ModernBERT-based token classifier)
	@if [ ! -f "models/halugate-detector/.downloaded" ] || [ ! -d "models/halugate-detector" ]; then \
		hf download KRLabsOrg/lettucedect-base-modernbert-en-v1 --local-dir models/halugate-detector && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/halugate-detector/.downloaded; \
	fi
	# NLI model for enhanced hallucination detection with explanations (ModernBERT-based)
	@if [ ! -f "models/ModernBERT-base-nli/.downloaded" ] || [ ! -d "models/ModernBERT-base-nli" ]; then \
		hf download tasksource/ModernBERT-base-nli --local-dir models/ModernBERT-base-nli && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/ModernBERT-base-nli/.downloaded; \
	fi
	# Fact-check classifier model (HaluGate Sentinel)
	@if [ ! -f "models/halugate-sentinel/.downloaded" ] || [ ! -d "models/halugate-sentinel" ]; then \
		hf download LLM-Semantic-Router/halugate-sentinel --local-dir models/halugate-sentinel && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/halugate-sentinel/.downloaded; \
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
	@if [ ! -f "models/Qwen3-Embedding-0.6B/.downloaded" ] || [ ! -d "models/Qwen3-Embedding-0.6B" ]; then \
		hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/Qwen3-Embedding-0.6B/.downloaded; \
	fi
	@if [ ! -f "models/embeddinggemma-300m/.downloaded" ] || [ ! -d "models/embeddinggemma-300m" ]; then \
		echo "Downloading google/embeddinggemma-300m (requires HF_TOKEN for gated model)..."; \
		hf download google/embeddinggemma-300m --local-dir models/embeddinggemma-300m && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/embeddinggemma-300m/.downloaded; \
	fi
	# Download hallucination mitigation models
	@if [ ! -f "models/halugate-detector/.downloaded" ] || [ ! -d "models/halugate-detector" ]; then \
		hf download KRLabsOrg/lettucedect-base-modernbert-en-v1 --local-dir models/halugate-detector && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/halugate-detector/.downloaded; \
	fi
	@if [ ! -f "models/ModernBERT-base-nli/.downloaded" ] || [ ! -d "models/ModernBERT-base-nli" ]; then \
		hf download tasksource/ModernBERT-base-nli --local-dir models/ModernBERT-base-nli && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/ModernBERT-base-nli/.downloaded; \
	fi
	@if [ ! -f "models/halugate-sentinel/.downloaded" ] || [ ! -d "models/halugate-sentinel" ]; then \
		hf download LLM-Semantic-Router/halugate-sentinel --local-dir models/halugate-sentinel && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/halugate-sentinel/.downloaded; \
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
	@if [ ! -f "models/Qwen3-Embedding-0.6B/.downloaded" ] || [ ! -d "models/Qwen3-Embedding-0.6B" ]; then \
		hf download Qwen/Qwen3-Embedding-0.6B --local-dir models/Qwen3-Embedding-0.6B && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/Qwen3-Embedding-0.6B/.downloaded; \
	fi

# Clean up minimal models to save disk space (for CI)
clean-minimal-models: ## Remove minimal models to save disk space
	@echo "Cleaning up minimal models to save disk space..."
	@rm -rf models/category_classifier_modernbert-base_model || true
	@rm -rf models/pii_classifier_modernbert-base_presidio_token_model || true
	@rm -rf models/jailbreak_classifier_modernbert-base_model || true
	@rm -rf models/pii_classifier_modernbert-base_model || true
	@echo "✓ Minimal models cleaned up"
