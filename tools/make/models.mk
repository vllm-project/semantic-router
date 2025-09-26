# ======== models.mk ========
# =  Everything For models  =
# ======== models.mk ========

download-models:
	@$(LOG_TARGET)
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

	@if [ ! -d "models/pii_classifier_modernbert_base_presidio_token_model" ]; then \
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
