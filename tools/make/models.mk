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
	@if [ ! -d "models/category_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/category_classifier_modernbert-base_model --local-dir models/category_classifier_modernbert-base_model; \
	fi
	@if [ ! -d "models/pii_classifier_modernbert-base_presidio_token_model" ]; then \
		hf download LLM-Semantic-Router/pii_classifier_modernbert-base_presidio_token_model --local-dir models/pii_classifier_modernbert-base_presidio_token_model; \
	fi
	@if [ ! -d "models/jailbreak_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model --local-dir models/jailbreak_classifier_modernbert-base_model; \
	fi
	@if [ ! -d "models/pii_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/pii_classifier_modernbert-base_model --local-dir models/pii_classifier_modernbert-base_model; \
	fi

# Full model set for local development and docs

download-models-full:
	@mkdir -p models
	@if [ ! -d "models/category_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/category_classifier_modernbert-base_model --local-dir models/category_classifier_modernbert-base_model; \
	fi
	@if [ ! -d "models/pii_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/pii_classifier_modernbert-base_model --local-dir models/pii_classifier_modernbert-base_model; \
	fi
	@if [ ! -d "models/jailbreak_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model --local-dir models/jailbreak_classifier_modernbert-base_model; \
	fi
	@if [ ! -d "models/pii_classifier_modernbert-base_presidio_token_model" ]; then \
		hf download LLM-Semantic-Router/pii_classifier_modernbert-base_presidio_token_model --local-dir models/pii_classifier_modernbert-base_presidio_token_model; \
	fi
	@if [ ! -d "models/lora_intent_classifier_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model --local-dir models/lora_intent_classifier_bert-base-uncased_model; \
	fi
	@if [ ! -d "models/lora_intent_classifier_roberta-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_intent_classifier_roberta-base_model --local-dir models/lora_intent_classifier_roberta-base_model; \
	fi
	@if [ ! -d "models/lora_intent_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_intent_classifier_modernbert-base_model --local-dir models/lora_intent_classifier_modernbert-base_model; \
	fi
	@if [ ! -d "models/lora_pii_detector_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model --local-dir models/lora_pii_detector_bert-base-uncased_model; \
	fi
	@if [ ! -d "models/lora_pii_detector_roberta-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_pii_detector_roberta-base_model --local-dir models/lora_pii_detector_roberta-base_model; \
	fi
	@if [ ! -d "models/lora_pii_detector_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_pii_detector_modernbert-base_model --local-dir models/lora_pii_detector_modernbert-base_model; \
	fi
	@if [ ! -d "models/lora_jailbreak_classifier_bert-base-uncased_model" ]; then \
		hf download LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model --local-dir models/lora_jailbreak_classifier_bert-base-uncased_model; \
	fi
	@if [ ! -d "models/lora_jailbreak_classifier_roberta-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_jailbreak_classifier_roberta-base_model --local-dir models/lora_jailbreak_classifier_roberta-base_model; \
	fi
	@if [ ! -d "models/lora_jailbreak_classifier_modernbert-base_model" ]; then \
		hf download LLM-Semantic-Router/lora_jailbreak_classifier_modernbert-base_model --local-dir models/lora_jailbreak_classifier_modernbert-base_model; \
	fi
