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
		Description:      "BERT-based domain/intent classifier with LoRA adapters for MMLU categories",
		ParameterSize:    "110M + LoRA",
		UsesLoRA:         true,
		NumClasses:       14, // MMLU categories
		MaxContextLength: 512,
		Tags:             []string{"classification", "lora", "mmlu", "domain", "bert"},
	},

	// PII Detection - BERT LoRA
	{
		LocalPath:        "models/mom-pii-classifier",
		RepoID:           "LLM-Semantic-Router/lora_pii_detector_bert-base-uncased_model",
		Aliases:          []string{"pii-detector", "pii-classifier", "privacy-guard", "lora_pii_detector_bert-base-uncased_model"},
		Purpose:          PurposePIIDetection,
		Description:      "BERT-based PII detector with LoRA adapters for 35 PII types",
		ParameterSize:    "110M + LoRA",
		UsesLoRA:         true,
		NumClasses:       35, // PII types
		MaxContextLength: 512,
		Tags:             []string{"pii", "privacy", "lora", "token-classification", "bert"},
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
		Description:      "ModernBERT-based jailbreak/prompt injection detector",
		ParameterSize:    "149M",
		UsesLoRA:         false,
		NumClasses:       2, // benign/jailbreak
		MaxContextLength: 512,
		Tags:             []string{"safety", "jailbreak", "prompt-injection", "modernbert"},
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

	// Embedding Models - mmBERT 2D Matryoshka (Multilingual)
	{
		LocalPath:        "models/mom-embedding-ultra",
		RepoID:           "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
		Aliases:          []string{"mmbert-embed-32k-2d-matryoshka", "mmbert-embedding", "embedding-mmbert", "mmbert", "embedding-ultra"},
		Purpose:          PurposeEmbedding,
		Description:      "ModernBERT 2D Matryoshka embedding: 307M params, 32K context, 1800+ languages (Glot500), STS 80.5 (exceeds Qwen3-0.6B), 1.6-3.1× faster than BGE-M3 with FA2",
		ParameterSize:    "307M",
		EmbeddingDim:     768, // Default, supports 512/256/128/64 via Matryoshka
		MaxContextLength: 32768,
		Tags:             []string{"embedding", "matryoshka", "2d-matryoshka", "multilingual", "modernbert", "long-context", "early-exit", "flash-attention-2"},
	},

	// ============================================================================
	// mmBERT-32K LoRA Models (32K context, YaRN RoPE scaling, multilingual)
	// Reference: https://huggingface.co/llm-semantic-router/mmbert-32k-yarn
	// ============================================================================

	// mmBERT-32K Intent Classifier
	{
		LocalPath:        "models/mmbert32k-intent-classifier-lora",
		RepoID:           "llm-semantic-router/mmbert32k-intent-classifier-lora",
		Aliases:          []string{"mmbert32k-intent", "mmbert-32k-intent", "intent-classifier-32k"},
		Purpose:          PurposeDomainClassification,
		Description:      "mmBERT-32K intent classifier with YaRN RoPE scaling for MMLU-Pro categories",
		ParameterSize:    "307M + LoRA",
		UsesLoRA:         true,
		NumClasses:       14, // MMLU-Pro categories
		MaxContextLength: 32768,
		Tags:             []string{"classification", "lora", "mmlu", "intent", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Fact-Check Classifier
	{
		LocalPath:        "models/mmbert32k-factcheck-classifier-lora",
		RepoID:           "llm-semantic-router/mmbert32k-factcheck-classifier-lora",
		Aliases:          []string{"mmbert32k-factcheck", "mmbert-32k-factcheck", "factcheck-classifier-32k", "fact-check-32k"},
		Purpose:          PurposeHallucinationSentinel,
		Description:      "mmBERT-32K fact-check classifier for determining if queries need verification",
		ParameterSize:    "307M + LoRA",
		UsesLoRA:         true,
		NumClasses:       2, // NO_FACT_CHECK_NEEDED / FACT_CHECK_NEEDED
		MaxContextLength: 32768,
		Tags:             []string{"factcheck", "lora", "mmbert-32k", "yarn", "multilingual", "rag"},
	},

	// mmBERT-32K Jailbreak Detector
	{
		LocalPath:        "models/mmbert32k-jailbreak-detector-lora",
		RepoID:           "llm-semantic-router/mmbert32k-jailbreak-detector-lora",
		Aliases:          []string{"mmbert32k-jailbreak", "mmbert-32k-jailbreak", "jailbreak-detector-32k", "prompt-guard-32k"},
		Purpose:          PurposeJailbreakDetection,
		Description:      "mmBERT-32K jailbreak/prompt injection detector with multilingual support",
		ParameterSize:    "307M + LoRA",
		UsesLoRA:         true,
		NumClasses:       2, // benign / jailbreak
		MaxContextLength: 32768,
		Tags:             []string{"safety", "jailbreak", "prompt-injection", "lora", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Feedback Detector (LoRA)
	{
		LocalPath:        "models/mmbert32k-feedback-detector-lora",
		RepoID:           "llm-semantic-router/mmbert32k-feedback-detector-lora",
		Aliases:          []string{"mmbert32k-feedback", "mmbert-32k-feedback", "feedback-detector-32k"},
		Purpose:          PurposeFeedbackDetection,
		Description:      "mmBERT-32K user feedback classifier for 4 satisfaction levels",
		ParameterSize:    "307M + LoRA",
		UsesLoRA:         true,
		NumClasses:       4, // SAT / NEED_CLARIFICATION / WRONG_ANSWER / WANT_DIFFERENT
		MaxContextLength: 32768,
		Tags:             []string{"feedback", "classification", "lora", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Feedback Detector (Merged - for Rust/Go inference)
	{
		LocalPath:        "models/mmbert32k-feedback-detector-merged",
		RepoID:           "llm-semantic-router/mmbert32k-feedback-detector-merged",
		Aliases:          []string{"mmbert32k-feedback-merged", "feedback-detector-32k-merged"},
		Purpose:          PurposeFeedbackDetection,
		Description:      "mmBERT-32K merged feedback detector (full model for Rust inference)",
		ParameterSize:    "307M",
		UsesLoRA:         false,
		NumClasses:       4, // SAT / NEED_CLARIFICATION / WRONG_ANSWER / WANT_DIFFERENT
		MaxContextLength: 32768,
		Tags:             []string{"feedback", "classification", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Intent Classifier (Merged)
	{
		LocalPath:        "models/mmbert32k-intent-classifier-merged",
		RepoID:           "llm-semantic-router/mmbert32k-intent-classifier-merged",
		Aliases:          []string{"mmbert32k-intent-merged", "intent-classifier-32k-merged"},
		Purpose:          PurposeDomainClassification,
		Description:      "mmBERT-32K merged intent classifier (full model for Rust inference)",
		ParameterSize:    "307M",
		UsesLoRA:         false,
		NumClasses:       14,
		MaxContextLength: 32768,
		Tags:             []string{"classification", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Fact-Check Classifier (Merged)
	{
		LocalPath:        "models/mmbert32k-factcheck-classifier-merged",
		RepoID:           "llm-semantic-router/mmbert32k-factcheck-classifier-merged",
		Aliases:          []string{"mmbert32k-factcheck-merged", "factcheck-classifier-32k-merged"},
		Purpose:          PurposeHallucinationSentinel,
		Description:      "mmBERT-32K merged fact-check classifier (full model for Rust inference)",
		ParameterSize:    "307M",
		UsesLoRA:         false,
		NumClasses:       2,
		MaxContextLength: 32768,
		Tags:             []string{"factcheck", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Jailbreak Detector (Merged)
	{
		LocalPath:        "models/mmbert32k-jailbreak-detector-merged",
		RepoID:           "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
		Aliases:          []string{"mmbert32k-jailbreak-merged", "jailbreak-detector-32k-merged"},
		Purpose:          PurposeJailbreakDetection,
		Description:      "mmBERT-32K merged jailbreak detector (full model for Rust inference)",
		ParameterSize:    "307M",
		UsesLoRA:         false,
		NumClasses:       2,
		MaxContextLength: 32768,
		Tags:             []string{"safety", "jailbreak", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K PII Detector (Merged)
	{
		LocalPath:        "models/mmbert32k-pii-detector-merged",
		RepoID:           "llm-semantic-router/mmbert32k-pii-detector-merged",
		Aliases:          []string{"mmbert32k-pii-merged", "pii-detector-32k-merged"},
		Purpose:          PurposePIIDetection,
		Description:      "mmBERT-32K merged PII detector (full model for Rust inference)",
		ParameterSize:    "307M",
		UsesLoRA:         false,
		NumClasses:       35,
		MaxContextLength: 32768,
		Tags:             []string{"pii", "privacy", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K PII Detector
	{
		LocalPath:        "models/mmbert32k-pii-detector-lora",
		RepoID:           "llm-semantic-router/mmbert32k-pii-detector-lora",
		Aliases:          []string{"mmbert32k-pii", "mmbert-32k-pii", "pii-detector-32k"},
		Purpose:          PurposePIIDetection,
		Description:      "mmBERT-32K PII detector for 17 entity types with BIO tagging",
		ParameterSize:    "307M + LoRA",
		UsesLoRA:         true,
		NumClasses:       35, // 17 entity types × 2 (B/I) + O
		MaxContextLength: 32768,
		Tags:             []string{"pii", "privacy", "token-classification", "lora", "mmbert-32k", "yarn", "multilingual"},
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

// ResolveModelPath resolves a model path or alias to its canonical local path
// This allows users to specify either:
// - Full path: "models/mom-embedding-pro"
// - Alias: "qwen3", "embedding-pro", etc.
//
// Returns the canonical LocalPath if found, or the original path if not in registry
func ResolveModelPath(path string) string {
	if path == "" {
		return ""
	}

	// Check if it's already a valid path in the registry
	if model := GetModelByPath(path); model != nil {
		return model.LocalPath
	}

	// Not found in registry, return as-is (might be a custom path)
	return path
}
