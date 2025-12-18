package config

// DefaultModelRegistry provides the default model registry mapping
// Users can override this by specifying mom_registry in their config.yaml
// If mom_registry is not specified in config.yaml, these defaults will be used
var DefaultModelRegistry = map[string]string{
	"models/mom-domain-classifier":    "LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model",
	"models/mom-pii-classifier":       "LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model",
	"models/mom-jailbreak-classifier": "LLM-Semantic-Router/lora_jailbreak_classifier_bert-base-uncased_model",
	"models/mom-halugate-detector":    "KRLabsOrg/lettucedect-base-modernbert-en-v1",
	"models/mom-halugate-sentinel":    "LLM-Semantic-Router/halugate-sentinel",
	"models/mom-halugate-explainer":   "tasksource/ModernBERT-base-nli",
	"models/mom-embedding-pro":        "Qwen/Qwen3-Embedding-0.6B",
	"models/mom-embedding-flash":      "google/embeddinggemma-300m",
}
