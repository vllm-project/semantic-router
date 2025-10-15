# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

# CI_MINIMAL_MODELS=true will download only the minimal set of models required for tests.
# Default behavior downloads the full set used for local development.

download-models:
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

download-models-minimal:
	@mkdir -p models
	# Pre-download tiny LLM for llm-katan (optional but speeds up first start)
	@if [ ! -f "models/Qwen/Qwen3-0.6B/.downloaded" ] || [ ! -d "models/Qwen/Qwen3-0.6B" ]; then \
		hf download Qwen/Qwen3-0.6B --local-dir models/Qwen/Qwen3-0.6B && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/Qwen/Qwen3-0.6B/.downloaded; \
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

# Full model set for local development and docs

download-models-full:
	@mkdir -p models
	# Pre-download tiny LLM for llm-katan (optional but speeds up first start)
	@if [ ! -f "models/Qwen/Qwen3-0.6B/.downloaded" ] || [ ! -d "models/Qwen/Qwen3-0.6B" ]; then \
		hf download Qwen/Qwen3-0.6B --local-dir models/Qwen/Qwen3-0.6B && printf '%s\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > models/Qwen/Qwen3-0.6B/.downloaded; \
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
