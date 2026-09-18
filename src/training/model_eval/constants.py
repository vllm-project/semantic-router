"""
Constants for served-model evaluation and explicit legacy MoM evaluation
All shared constants, registries and configs are defined here.
"""

from copy import deepcopy

# Historical reference only; loaders must use the artifact's declared base.
BASE_MODEL_ID = "jhu-clsp/mmBERT-base"

# Supported langs
LANGUAGE_CODES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "hi": "Hindi",
    "pt": "Portuguese",
    "it": "Italian",
}

# Registry of all MoM models (merged as well as LoRA)
LEGACY_MODEL_REGISTRY = {
    "feedback": {
        "id": "llm-semantic-router/mmbert32k-feedback-detector-merged",
        "lora_id": "llm-semantic-router/mmbert32k-feedback-detector-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/feedback-detector-dataset",
        "labels": ["SAT", "NEED_CLARIFICATION", "WRONG_ANSWER", "WANT_DIFFERENT"],
        "text_col": "text",
        "label_col": "label",
        "split": "validation",
    },
    "jailbreak": {
        "id": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
        "lora_id": "llm-semantic-router/mmbert32k-jailbreak-detector-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/jailbreak-detection-dataset",
        "labels": ["benign", "jailbreak"],
        "dataset_label_aliases": {"safe": "benign", "unsafe": "jailbreak"},
        "text_col": "text",
        "label_col": "label",
        "split": "test",
    },
    "fact-check": {
        "id": "llm-semantic-router/mmbert32k-factcheck-classifier-merged",
        "lora_id": "llm-semantic-router/mmbert32k-factcheck-classifier-lora",
        "type": "text_classification",
        "hf_dataset": "llm-semantic-router/fact-check-classification-dataset",
        "labels": ["NO_FACT_CHECK_NEEDED", "FACT_CHECK_NEEDED"],
        "text_col": "text",
        "label_col": "label_id",
        "split": "test",
    },
    "intent": {
        "id": "llm-semantic-router/mmbert32k-intent-classifier-merged",
        "lora_id": "llm-semantic-router/mmbert32k-intent-classifier-lora",
        "type": "text_classification",
        "hf_dataset": "TIGER-Lab/MMLU-Pro",
        "labels": [
            "biology",
            "business",
            "chemistry",
            "computer science",
            "economics",
            "engineering",
            "health",
            "history",
            "law",
            "math",
            "other",
            "philosophy",
            "physics",
            "psychology",
        ],
        "text_col": "question",
        "label_col": "category",
        "split": "test",
    },
    "pii": {
        "id": "llm-semantic-router/mmbert32k-pii-detector-merged",
        "lora_id": "llm-semantic-router/mmbert32k-pii-detector-lora",
        "type": "token_classification",
        "hf_dataset": "presidio",
        "labels": [
            "O",
            "B-AGE",
            "I-AGE",
            "B-CREDIT_CARD",
            "I-CREDIT_CARD",
            "B-DATE_TIME",
            "I-DATE_TIME",
            "B-DOMAIN_NAME",
            "I-DOMAIN_NAME",
            "B-EMAIL_ADDRESS",
            "I-EMAIL_ADDRESS",
            "B-GPE",
            "I-GPE",
            "B-IBAN_CODE",
            "I-IBAN_CODE",
            "B-IP_ADDRESS",
            "I-IP_ADDRESS",
            "B-NRP",
            "I-NRP",
            "B-ORGANIZATION",
            "I-ORGANIZATION",
            "B-PERSON",
            "I-PERSON",
            "B-PHONE_NUMBER",
            "I-PHONE_NUMBER",
            "B-STREET_ADDRESS",
            "I-STREET_ADDRESS",
            "B-TITLE",
            "I-TITLE",
            "B-US_DRIVER_LICENSE",
            "I-US_DRIVER_LICENSE",
            "B-US_SSN",
            "I-US_SSN",
            "B-ZIP_CODE",
            "I-ZIP_CODE",
        ],
        "text_col": "tokens",
        "label_col": "labels",
        "split": "test",
    },
}

# Immutable native releases. Keep these in sync with config/registry.go; tests
# compare every entry, including releases outside this classifier evaluator.
VELA_RELEASE_REVISIONS = {
    "llm-semantic-router/Vela-1.0-Encoder-307M": "fe9ccc074b781bc0e2e13c2c8d26f2640410636a",
    "llm-semantic-router/Vela-1.0-Encoder-307M-FactCheck": "99ede1aba1563e59e416f744d25b3f6b7e9d8274",
    "llm-semantic-router/Vela-1.0-Encoder-307M-Domain": "f6354f54adcf38770f635ad903be2b00577f6c11",
    "llm-semantic-router/Vela-1.0-Encoder-307M-PII": "6d3300c4bd7975f30a664503f6c725cf1fbbad48",
    "llm-semantic-router/Vela-1.0-Encoder-307M-Modality": "5384b8997e3cbb79ca3a670e869577f4e4f4997e",
    "llm-semantic-router/Vela-1.0-Encoder-307M-Feedback": "47434a7fd7c245c0c7c17564a000b3c56ccfec41",
    "llm-semantic-router/Vela-1.0-Encoder-307M-Reranker": "a388e41cbbd5dc5f16b6389fa76d0b8b8a38a8bf",
    "llm-semantic-router/Vela-1.0-Encoder-307M-Hazard": "5dd25f2cc3c98f338e6a79b667662d60f936a28d",
    "llm-semantic-router/Vela-1.0-Encoder-307M-Safety": "6e70e725a5f4d86da10f5be5e4dfd1da0358bb85",
    "llm-semantic-router/Vela-1.0-Encoder-307M-Guard": "087f9e401012df839c83717b746967ac7aebfa3e",
    "llm-semantic-router/Vela-1.0-Encoder-307M-Embedding": "1e57cebf5a7b7fec6e6973f05bbca97c5cca4436",
}

MODEL_REGISTRY = deepcopy(LEGACY_MODEL_REGISTRY)
for _role, _suffix in {
    "feedback": "Feedback",
    "jailbreak": "Guard",
    "fact-check": "FactCheck",
    "intent": "Domain",
    "pii": "PII",
}.items():
    _entry = MODEL_REGISTRY[_role]
    _entry["id"] = f"llm-semantic-router/Vela-1.0-Encoder-307M-{_suffix}"
    _entry["revision"] = VELA_RELEASE_REVISIONS[_entry["id"]]
    # Vela releases are self-contained task models.
    del _entry["lora_id"]
MODEL_REGISTRY["feedback"]["labels"].append("NO_FEEDBACK")
# Guard detects instruction attacks. The historical dataset mixed toxicity
# with attacks; identical binary label names do not make that gold compatible.
del MODEL_REGISTRY["jailbreak"]["hf_dataset"]
del MODEL_REGISTRY["jailbreak"]["dataset_label_aliases"]

COLLECTIONS = {"served": MODEL_REGISTRY, "legacy-mom": LEGACY_MODEL_REGISTRY}


def model_registry(collection="served"):
    """Return an explicit collection; never fall back to a legacy artifact."""
    if collection not in COLLECTIONS:
        raise ValueError(f"Unknown model collection: {collection}")
    return COLLECTIONS[collection]
