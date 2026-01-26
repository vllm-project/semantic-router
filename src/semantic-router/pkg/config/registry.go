package config

import "strings"

// ModelPurpose describes what the model is used for
type ModelPurpose string

const (
	PurposeDomainClassification   ModelPurpose = "domain-classification"   // Classify text into domains/categories
	PurposePIIDetection           ModelPurpose = "pii-detection"           // Detect personally identifiable information
	PurposeJailbreakDetection     ModelPurpose = "jailbreak-detection"     // Detect prompt injection/jailbreak attempts
	PurposeHallucinationSentinel  ModelPurpose = "hallucination-sentinel"  // Detect potential hallucinations
	PurposeHallucinationDetector  ModelPurpose = "hallucination-detector"  // Verify factual accuracy
	PurposeHallucinationExplainer ModelPurpose = "hallucination-explainer" // Explain hallucination reasoning
	PurposeFeedbackDetection      ModelPurpose = "feedback-detection"      // Detect user feedback type
	PurposeEmbedding              ModelPurpose = "embedding"               // Generate text embeddings
	PurposeSemanticSimilarity     ModelPurpose = "semantic-similarity"     // Compute semantic similarity
)

// ModelSpec defines a model's metadata and capabilities
type ModelSpec struct {
	// Primary local path (canonical name)
	LocalPath string `json:"local_path" yaml:"local_path"`

	// HuggingFace repository ID
	RepoID string `json:"repo_id" yaml:"repo_id"`

	// Alternative names/aliases for this model
	Aliases []string `json:"aliases,omitempty" yaml:"aliases,omitempty"`

	// Primary purpose of this model
	Purpose ModelPurpose `json:"purpose" yaml:"purpose"`

	// Human-readable description
	Description string `json:"description" yaml:"description"`

	// Model size in parameters (e.g., "33M", "600M")
	ParameterSize string `json:"parameter_size,omitempty" yaml:"parameter_size,omitempty"`

	// Embedding dimension (for embedding models)
	EmbeddingDim int `json:"embedding_dim,omitempty" yaml:"embedding_dim,omitempty"`

	// Maximum context length
	MaxContextLength int `json:"max_context_length,omitempty" yaml:"max_context_length,omitempty"`

	// Whether this model uses LoRA adapters
	UsesLoRA bool `json:"uses_lora,omitempty" yaml:"uses_lora,omitempty"`

	// Number of classification classes (for classifiers)
	NumClasses int `json:"num_classes,omitempty" yaml:"num_classes,omitempty"`

	// Additional tags for filtering/searching
	Tags []string `json:"tags,omitempty" yaml:"tags,omitempty"`
}

// DefaultModelRegistry provides the structured model registry
// Users can override this by specifying mom_registry in their config.yaml
var DefaultModelRegistry = []ModelSpec{
	// Domain/Intent Classification
	{
		LocalPath:        "models/mom-domain-classifier",
		RepoID:           "LLM-Semantic-Router/lora_intent_classifier_bert-base-uncased_model",
		Aliases:          []string{"domain-classifier", "intent-classifier", "category-classifier", "category_classifier_modernbert-base_model", "lora_intent_classifier_bert-base-uncased_model"},
		Purpose:          PurposeDomainClassification,
		Description:      "Domain/intent classifier compatible with ModernBERT-base-32k for 32K context support. Uses traditional classifier weights (compatible with ModernBERT-base-32k base model) for MMLU categories",
		ParameterSize:    "149M (ModernBERT-base-32k) + classifier",
		UsesLoRA:         true,
		NumClasses:       14, // MMLU categories
		MaxContextLength: 32768, // Supports 32K tokens via ModernBERT-base-32k
		Tags:             []string{"classification", "lora", "mmlu", "domain", "modernbert", "long-context"},
	},

	// PII Detection - BERT LoRA
	{
		LocalPath:        "models/mom-pii-classifier",
		RepoID:           "LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model",
		Aliases:          []string{"pii-detector", "pii-classifier", "privacy-guard", "lora_pii_detector_bert-base-uncased_model"},
		Purpose:          PurposePIIDetection,
		Description:      "PII detector compatible with ModernBERT-base-32k for 32K context support. Uses traditional classifier weights (compatible with ModernBERT-base-32k base model) for 35 PII types",
		ParameterSize:    "149M (ModernBERT-base-32k) + classifier",
		UsesLoRA:         true,
		NumClasses:       35, // PII types
		MaxContextLength: 32768, // Supports 32K tokens via ModernBERT-base-32k
		Tags:             []string{"pii", "privacy", "lora", "token-classification", "modernbert", "long-context"},
	},

	// PII Detection - ModernBERT (Token-level)
	{
		LocalPath:        "models/mom-mmbert-pii-detector",
		RepoID:           "llm-semantic-router/mmbert-pii-detector-merged",
		Aliases:          []string{"mmbert-pii-detector", "mmbert-pii-detector-merged", "pii_classifier_modernbert-base_presidio_token_model", "pii_classifier_modernbert-base_model", "pii_classifier_modernbert_model", "pii_classifier_modernbert_ai4privacy_token_model"},
		Purpose:          PurposePIIDetection,
		Description:      "ModernBERT-based merged PII detector for token-level classification",
		ParameterSize:    "149M",
		UsesLoRA:         false,
		NumClasses:       35, // PII types
		MaxContextLength: 8192,
		Tags:             []string{"pii", "privacy", "modernbert", "token-classification", "merged"},
	},

	// Jailbreak Detection
	{
		LocalPath:        "models/mom-jailbreak-classifier",
		RepoID:           "LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model",
		Aliases:          []string{"jailbreak-detector", "prompt-guard", "safety-classifier", "jailbreak_classifier_modernbert-base_model", "lora_jailbreak_classifier_bert-base-uncased_model", "jailbreak_classifier_modernbert_model"},
		Purpose:          PurposeJailbreakDetection,
		Description:      "ModernBERT-based jailbreak/prompt injection detector compatible with ModernBERT-base-32k for 32K context support",
		ParameterSize:    "149M (ModernBERT-base-32k)",
		UsesLoRA:         false,
		NumClasses:       2, // benign/jailbreak
		MaxContextLength: 32768, // Supports 32K tokens via ModernBERT-base-32k
		Tags:             []string{"safety", "jailbreak", "prompt-injection", "modernbert", "long-context"},
	},

	// Hallucination Detection - Sentinel
	{
		LocalPath:        "models/mom-halugate-sentinel",
		RepoID:           "LLM-Semantic-Router/halugate-sentinel",
		Aliases:          []string{"hallucination-sentinel", "halugate-sentinel"},
		Purpose:          PurposeHallucinationSentinel,
		Description:      "First-stage hallucination detection sentinel for fast screening",
		ParameterSize:    "110M",
		NumClasses:       2, // hallucination/no-hallucination
		MaxContextLength: 512,
		Tags:             []string{"hallucination", "sentinel", "screening", "bert"},
	},

	// Hallucination Detection - Detector
	{
		LocalPath:        "models/mom-halugate-detector",
		RepoID:           "KRLabsOrg/lettucedect-base-modernbert-en-v1",
		Aliases:          []string{"hallucination-detector", "halugate-detector", "lettucedect"},
		Purpose:          PurposeHallucinationDetector,
		Description:      "ModernBERT-based hallucination detector for accurate verification",
		ParameterSize:    "149M",
		EmbeddingDim:     768,
		MaxContextLength: 8192, // ModernBERT supports long context
		Tags:             []string{"hallucination", "modernbert", "verification"},
	},

	// Hallucination Detection - Explainer
	{
		LocalPath:        "models/mom-halugate-explainer",
		RepoID:           "tasksource/ModernBERT-base-nli",
		Aliases:          []string{"hallucination-explainer", "halugate-explainer", "nli-explainer"},
		Purpose:          PurposeHallucinationExplainer,
		Description:      "ModernBERT NLI model for explaining hallucination reasoning",
		ParameterSize:    "149M",
		NumClasses:       3, // entailment/neutral/contradiction
		MaxContextLength: 8192,
		Tags:             []string{"hallucination", "nli", "explainability", "modernbert"},
	},

	// Feedback Detection
	{
		LocalPath:        "models/mom-feedback-detector",
		RepoID:           "llm-semantic-router/feedback-detector",
		Aliases:          []string{"feedback-detector", "user-feedback-classifier"},
		Purpose:          PurposeFeedbackDetection,
		Description:      "ModernBERT-based user feedback classifier for 4 feedback types",
		ParameterSize:    "149M",
		NumClasses:       4, // satisfied/need_clarification/wrong_answer/want_different
		MaxContextLength: 8192,
		Tags:             []string{"feedback", "classification", "modernbert", "user-intent"},
	},

	// Embedding Models - Pro (High Quality)
	{
		LocalPath:        "models/mom-embedding-pro",
		RepoID:           "Qwen/Qwen3-Embedding-0.6B",
		Aliases:          []string{"Qwen3-Embedding-0.6B", "embedding-pro", "qwen3"},
		Purpose:          PurposeEmbedding,
		Description:      "High-quality embedding model with 32K context support",
		ParameterSize:    "600M",
		EmbeddingDim:     1024,
		MaxContextLength: 32768,
		Tags:             []string{"embedding", "long-context", "qwen", "high-quality"},
	},

	// Embedding Models - Flash (Balanced)
	{
		LocalPath:        "models/mom-embedding-flash",
		RepoID:           "google/embeddinggemma-300m",
		Aliases:          []string{"embeddinggemma-300m", "embedding-flash", "gemma"},
		Purpose:          PurposeEmbedding,
		Description:      "Fast embedding model with Matryoshka support (768/512/256/128 dims)",
		ParameterSize:    "300M",
		EmbeddingDim:     768, // Default, supports 512/256/128 via Matryoshka
		MaxContextLength: 2048,
		Tags:             []string{"embedding", "matryoshka", "gemma", "fast", "multilingual"},
	},

	// Embedding Models - Light (Fast)
	{
		LocalPath:        "models/mom-embedding-light",
		RepoID:           "sentence-transformers/all-MiniLM-L12-v2",
		Aliases:          []string{"all-MiniLM-L12-v2", "embedding-light", "bert-light"},
		Purpose:          PurposeSemanticSimilarity,
		Description:      "Lightweight sentence transformer for fast semantic similarity",
		ParameterSize:    "33M",
		EmbeddingDim:     384,
		MaxContextLength: 512,
		Tags:             []string{"embedding", "sentence-transformer", "fast", "lightweight"},
	},
}

// GetModelByPath returns a model spec by its local path or alias
func GetModelByPath(path string) *ModelSpec {
	for i := range DefaultModelRegistry {
		model := &DefaultModelRegistry[i]
		// Check primary path
		if model.LocalPath == path {
			return model
		}
		// Check aliases
		for _, alias := range model.Aliases {
			if alias == path || "models/"+alias == path {
				return model
			}
		}
	}
	return nil
}

// GetModelsByPurpose returns all models for a specific purpose
func GetModelsByPurpose(purpose ModelPurpose) []ModelSpec {
	var models []ModelSpec
	for _, model := range DefaultModelRegistry {
		if model.Purpose == purpose {
			models = append(models, model)
		}
	}
	return models
}

// GetModelsByTag returns all models with a specific tag
func GetModelsByTag(tag string) []ModelSpec {
	var models []ModelSpec
	for _, model := range DefaultModelRegistry {
		for _, t := range model.Tags {
			if t == tag {
				models = append(models, model)
				break
			}
		}
	}
	return models
}

// ToLegacyRegistry converts the structured registry to the legacy map format
// This maintains backward compatibility with existing code
// It includes both the primary LocalPath and all aliases
func ToLegacyRegistry() map[string]string {
	legacy := make(map[string]string)
	for _, model := range DefaultModelRegistry {
		// Add primary path
		legacy[model.LocalPath] = model.RepoID

		// Add all aliases (with and without "models/" prefix)
		for _, alias := range model.Aliases {
			// Add alias as-is
			legacy[alias] = model.RepoID
			// Add alias with "models/" prefix if not already present
			if !strings.HasPrefix(alias, "models/") {
				legacy["models/"+alias] = model.RepoID
			}
		}
	}
	return legacy
}
