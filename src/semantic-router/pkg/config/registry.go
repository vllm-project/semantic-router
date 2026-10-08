package config

import "strings"

// ModelPurpose describes what the model is used for
type ModelPurpose string

const (
	PurposeEncoder               ModelPurpose = "encoder"                // Base encoder for task adaptation
	PurposeDomainClassification  ModelPurpose = "domain-classification"  // Classify text into domains/categories
	PurposePIIDetection          ModelPurpose = "pii-detection"          // Detect personally identifiable information
	PurposeJailbreakDetection    ModelPurpose = "jailbreak-detection"    // Detect prompt injection/jailbreak attempts
	PurposeHallucinationSentinel ModelPurpose = "hallucination-sentinel" // Detect potential hallucinations
	PurposeHallucinationDetector ModelPurpose = "hallucination-detector" // Verify factual accuracy
	PurposeFeedbackDetection     ModelPurpose = "feedback-detection"     // Detect user feedback type
	PurposeModalityDetection     ModelPurpose = "modality-detection"     // Classify prompts into text/image/both modalities
	PurposeEmbedding             ModelPurpose = "embedding"              // Generate text embeddings
	PurposeSafety                ModelPurpose = "safety"                 // Detect unsafe content
	PurposeHazard                ModelPurpose = "hazard"                 // Identify independent content hazards
	PurposeReranking             ModelPurpose = "reranking"              // Rank query-document pairs
	PurposeSemanticSimilarity    ModelPurpose = "semantic-similarity"    // Compute semantic similarity
	PurposeRoutingSignals        ModelPurpose = "routing-signals"        // Answer several built-in routing signals in one call
)

// Vela2SignalModel is the registry path of Vela 2.0 0.3B, the default model
// of every built-in signal it answers.
const Vela2SignalModel = "models/Vela-2.0-0.3B"

// Registry paths of the larger Vela 2.0 sizes a decision model may name.
const (
	Vela2Model08B = "models/Vela-2.0-0.8B"
	Vela2Model4B  = "models/Vela-2.0-4B"
	Vela2Model9B  = "models/Vela-2.0-9B"
)

// ModelSpec defines a model's metadata and capabilities
type ModelSpec struct {
	// Primary local path (canonical name)
	LocalPath string `json:"local_path" yaml:"local_path"`

	// HuggingFace repository ID
	RepoID string `json:"repo_id" yaml:"repo_id"`

	// Immutable release revision, when the built-in artifact is pinned.
	Revision string `json:"revision,omitempty" yaml:"revision,omitempty"`

	// Built-in artifact download policy; this is not a user configuration field.
	// Only applied when the resolved repository still matches this entry.
	DownloadExcludePatterns []string `json:"-" yaml:"-"`

	// RuntimeProvisioned marks a release the model runtime downloads and
	// verifies itself; the router provisions nothing for it.
	RuntimeProvisioned bool `json:"-" yaml:"-"`

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

	// Maximum context length that this model's weights were trained on.
	// This is the safe maximum - sending more tokens may result in degraded performance.
	MaxContextLength int `json:"max_context_length,omitempty" yaml:"max_context_length,omitempty"`

	// Base model maximum context length (for LoRA/classifier models).
	// If set, indicates the base model can handle this context length, but classifier/LoRA
	// weights were only trained on MaxContextLength. Use with caution beyond MaxContextLength.
	// Example: BaseModelMaxContext=32768, MaxContextLength=512 means base model supports 32K
	// but classifier weights were trained on 512 tokens.
	BaseModelMaxContext int `json:"base_model_max_context,omitempty" yaml:"base_model_max_context,omitempty"`

	// Whether this model uses LoRA adapters
	UsesLoRA bool `json:"uses_lora,omitempty" yaml:"uses_lora,omitempty"`

	// DefaultAdapter declares task semantics for implicit built-in bindings.
	// Explicit recipe bindings always take precedence.
	DefaultAdapter string `json:"default_adapter,omitempty" yaml:"default_adapter,omitempty"`

	// SharedDeployment marks a model that answers several modules' signals:
	// every module that names it on one device runs one implicit deployment,
	// so the model loads once and a request's questions share one call.
	SharedDeployment bool `json:"-" yaml:"-"`

	// CPUProfile is the model_runtime profile an implicit CPU deployment of
	// the model runs; the runtime's accuracy record for the model backs it.
	CPUProfile string `json:"-" yaml:"-"`

	// RequiresGPU marks a model whose implicit deployment runs on a GPU only:
	// a module that names it runs it on the best GPU, and a host without one
	// cannot serve it.
	RequiresGPU bool `json:"-" yaml:"-"`

	// Number of classification classes (for classifiers)
	NumClasses int `json:"num_classes,omitempty" yaml:"num_classes,omitempty"`

	// Additional tags for filtering/searching
	Tags []string `json:"tags,omitempty" yaml:"tags,omitempty"`
}

var velaTrainingArtifactPatterns = []string{"reproduction/*", "reproducibility/*", "lora/*"}

// velaShieldArtifactPatterns keep a Shield download at the root sequence
// classifier. The repository also publishes auxiliary heads, a separate
// label-conditioned encoder and demo files that the router does not load.
// Patterns follow HF fnmatch semantics, where '*' crosses directories.
var velaShieldArtifactPatterns = append([]string{"lc/*", "heads/*", "demo.py", "DEMO_OUTPUT.txt"}, velaTrainingArtifactPatterns...)

// DefaultModelRegistry provides the structured model registry
// Users can override this by specifying mom_registry in their config.yaml
var DefaultModelRegistry = []ModelSpec{
	// Vela 2.0 0.3B answers the built-in domain, Guard, safety, fact-check,
	// feedback and modality signals as questions, and PII and hallucination
	// with its span presets. The model runtime downloads and verifies it; on
	// CPU, max_speed runs its float32-packed copy, which keeps its answers
	// (src/model-runtime/docs/records/vela2-parity.md).
	{
		LocalPath:          Vela2SignalModel,
		RepoID:             "vllm-sr/Vela-2.0-0.3B",
		Revision:           "a3209a50dc3ebd7e3b7520440d8fba666000f4c4",
		Aliases:            []string{"Vela-2.0-0.3B"},
		Purpose:            PurposeRoutingSignals,
		Description:        "Answer the built-in routing signals in one call: domain, prompt attacks, safety, fact-check need, feedback and modality, with PII and hallucination spans. Supports up to 8K input.",
		ParameterSize:      "309M encoder",
		MaxContextLength:   8192,
		RuntimeProvisioned: true,
		SharedDeployment:   true,
		CPUProfile:         "max_speed",
		Tags:               []string{"vela", "vela2", "multi-task", "spans", "multilingual"},
	},
	// The larger Vela 2.0 sizes answer the same questions and span presets as
	// the 0.3B; global.model_catalog.system.decision_model selects one. They
	// are Qwen3.5 hybrid decoders: the 0.8B runs on a CPU at seconds per
	// request, and the 4B and 9B run on a GPU only.
	{
		LocalPath:          Vela2Model08B,
		RepoID:             "vllm-sr/Vela-2.0-0.8B",
		Revision:           "a778eb2ae2304cfa72fca7e53a19136dea5be012",
		Aliases:            []string{"Vela-2.0-0.8B"},
		Purpose:            PurposeRoutingSignals,
		Description:        "Answer the built-in routing signals in one call, with PII and hallucination spans. Supports up to 16K input.",
		ParameterSize:      "756M decoder",
		MaxContextLength:   16384,
		RuntimeProvisioned: true,
		SharedDeployment:   true,
		Tags:               []string{"vela", "vela2", "multi-task", "spans", "multilingual"},
	},
	{
		LocalPath:          Vela2Model4B,
		RepoID:             "vllm-sr/Vela-2.0-4B",
		Revision:           "c1e64d4f872cb38bc58502e6888340100bab9d55",
		Aliases:            []string{"Vela-2.0-4B"},
		Purpose:            PurposeRoutingSignals,
		Description:        "Answer the built-in routing signals in one call, with PII and hallucination spans, on a GPU. Supports up to 16K input.",
		ParameterSize:      "4.2B decoder",
		MaxContextLength:   16384,
		RuntimeProvisioned: true,
		SharedDeployment:   true,
		RequiresGPU:        true,
		Tags:               []string{"vela", "vela2", "multi-task", "spans", "multilingual"},
	},
	{
		LocalPath:          Vela2Model9B,
		RepoID:             "vllm-sr/Vela-2.0-9B",
		Revision:           "bc8761637d8788619dfbaf6d8890128efe85fd40",
		Aliases:            []string{"Vela-2.0-9B"},
		Purpose:            PurposeRoutingSignals,
		Description:        "Answer the built-in routing signals in one call, with PII and hallucination spans, on a GPU. Supports up to 16K input.",
		ParameterSize:      "7.9B decoder",
		MaxContextLength:   16384,
		RuntimeProvisioned: true,
		SharedDeployment:   true,
		RequiresGPU:        true,
		Tags:               []string{"vela", "vela2", "multi-task", "spans", "multilingual"},
	},
	// Vela releases use immutable revisions. Legacy aliases below retain their
	// original repositories so an explicit old configuration stays reproducible.
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M",
		Revision:                "fe9ccc074b781bc0e2e13c2c8d26f2640410636a",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M"},
		Purpose:                 PurposeEncoder,
		Description:             "Build specialized, multilingual routing capabilities with up to 32K context. Use the Embedding model for search and retrieval.",
		ParameterSize:           "307M encoder",
		EmbeddingDim:            768,
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "encoder", "multilingual", "long-context"},
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-FactCheck",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-FactCheck",
		Revision:                "99ede1aba1563e59e416f744d25b3f6b7e9d8274",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-FactCheck"},
		Purpose:                 PurposeHallucinationSentinel,
		Description:             "Identify requests that need factual verification, with up to 32K input. It does not verify factual claims.",
		ParameterSize:           "307M encoder + classifier",
		NumClasses:              2,
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "factcheck", "classification", "multilingual", "long-context"},
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Domain",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Domain",
		Revision:                "f6354f54adcf38770f635ad903be2b00577f6c11",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Domain"},
		Purpose:                 PurposeDomainClassification,
		Description:             "Identify a request's subject across 14 domains for routing to relevant expertise. Supports up to 32K input.",
		ParameterSize:           "307M encoder + classifier",
		NumClasses:              14,
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "domain", "classification", "merged", "multilingual", "long-context"},
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-PII",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-PII",
		Revision:                "6d3300c4bd7975f30a664503f6c725cf1fbbad48",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-PII"},
		Purpose:                 PurposePIIDetection,
		Description:             "Locate personal information across 17 entity types with 35 BIO labels. Supports up to 32K input, with overlapping windows for long scans.",
		ParameterSize:           "307M encoder + classifier",
		NumClasses:              35,
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "pii", "privacy", "token-classification", "merged", "multilingual", "long-context"},
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Modality",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Modality",
		Revision:                "5384b8997e3cbb79ca3a670e869577f4e4f4997e",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Modality"},
		Purpose:                 PurposeModalityDetection,
		Description:             "Classify written requests into text generation, image generation, or both. This is a text classifier. Supports up to 32K input.",
		ParameterSize:           "307M encoder + classifier",
		NumClasses:              3,
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "modality", "text-only", "classification", "merged", "multilingual", "long-context"},
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Feedback",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Feedback",
		Revision:                "47434a7fd7c245c0c7c17564a000b3c56ccfec41",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Feedback"},
		Purpose:                 PurposeFeedbackDetection,
		Description:             "Recognize satisfaction, clarification, corrections, alternative requests, and messages without feedback. Supports up to 32K input.",
		ParameterSize:           "307M encoder + classifier",
		NumClasses:              5,
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "feedback", "classification", "merged", "multilingual", "long-context"},
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Embedding",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Embedding",
		Revision:                "1e57cebf5a7b7fec6e6973f05bbca97c5cca4436",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Embedding"},
		Purpose:                 PurposeEmbedding,
		Description:             "Find relevant multilingual context with flexible embedding dimensions and encoder depths. Supports up to 32K input; retrieval quality varies with representation size.",
		ParameterSize:           "307M encoder",
		EmbeddingDim:            768,
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "embedding", "multilingual", "long-context", "2d-matryoshka", "early-exit"},
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Guard",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Guard",
		Revision:                "087f9e401012df839c83717b746967ac7aebfa3e",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Guard"},
		Purpose:                 PurposeJailbreakDetection,
		Description:             "Detect prompt injection and jailbreak attacks across multilingual requests.",
		ParameterSize:           "307M encoder + classifier",
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "multilingual", "long-context", "guard", "prompt-injection", "classification"},
		NumClasses:              2,
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Safety",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Safety",
		Revision:                "6e70e725a5f4d86da10f5be5e4dfd1da0358bb85",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Safety"},
		Purpose:                 PurposeSafety,
		Description:             "Identify unsafe content independently of prompt attacks.",
		ParameterSize:           "307M encoder + classifier",
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "multilingual", "long-context", "safety", "classification"},
		NumClasses:              2,
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Shield",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Shield",
		Revision:                "a981a99eeb05a2859b88b5cee9af4352897ec4ec",
		DownloadExcludePatterns: velaShieldArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Shield"},
		Purpose:                 PurposeSafety,
		Description:             "Identify unsafe requests with a jointly trained multilingual safety encoder; alternative to Vela Safety.",
		ParameterSize:           "307M encoder + classifier",
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "multilingual", "long-context", "safety", "classification"},
		NumClasses:              2,
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Hazard",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Hazard",
		Revision:                "5dd25f2cc3c98f338e6a79b667662d60f936a28d",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Hazard"},
		Purpose:                 PurposeHazard,
		Description:             "Identify 12 independent content hazards using the published operating point and overlapping windows.",
		ParameterSize:           "307M encoder + classifier",
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "multilingual", "long-context", "hazard", "multi-label-classification"},
		NumClasses:              12,
	},
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Reranker",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Reranker",
		Revision:                "a388e41cbbd5dc5f16b6389fa76d0b8b8a38a8bf",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Reranker"},
		Purpose:                 PurposeReranking,
		Description:             "Rank multilingual query-document pairs with selectable encoder depth and representation size.",
		ParameterSize:           "307M encoder + reranker",
		MaxContextLength:        32768,
		Tags:                    []string{"vela", "multilingual", "long-context", "reranker", "2d-matryoshka", "early-exit"},
	},
	// Domain/Intent Classification
	{
		LocalPath:           "models/mom-domain-classifier",
		RepoID:              "vllm-sr/lora_intent_classifier_bert-base-uncased_model",
		Aliases:             []string{"domain-classifier", "intent-classifier", "category-classifier", "category_classifier_modernbert-base_model", "lora_intent_classifier_bert-base-uncased_model"},
		Purpose:             PurposeDomainClassification,
		Description:         "Domain/intent classifier using BERT-base-uncased LoRA adapter. Can be used with ModernBERT-base-32k base model for extended context, but LoRA weights were trained on 512-token context.",
		ParameterSize:       "149M (ModernBERT-base-32k) + classifier",
		UsesLoRA:            true,
		NumClasses:          14,    // MMLU categories
		MaxContextLength:    512,   // LoRA weights trained on 512 tokens - safe maximum
		BaseModelMaxContext: 32768, // Base model (ModernBERT-base-32k) supports 32K, but use with caution beyond MaxContextLength
		Tags:                []string{"classification", "lora", "mmlu", "domain", "modernbert"},
	},

	// PII Detection - BERT LoRA
	{
		LocalPath:           "models/mom-pii-classifier",
		RepoID:              "vllm-sr/lora_pii_detector_bert-base-uncased_model",
		Aliases:             []string{"pii-detector", "pii-classifier", "privacy-guard", "lora_pii_detector_bert-base-uncased_model"},
		Purpose:             PurposePIIDetection,
		Description:         "PII detector using BERT-base-uncased LoRA adapter. Can be used with ModernBERT-base-32k base model for extended context, but LoRA weights were trained on 512-token context.",
		ParameterSize:       "149M (ModernBERT-base-32k) + classifier",
		UsesLoRA:            true,
		NumClasses:          35,    // PII types
		MaxContextLength:    512,   // LoRA weights trained on 512 tokens - safe maximum
		BaseModelMaxContext: 32768, // Base model (ModernBERT-base-32k) supports 32K, but use with caution beyond MaxContextLength
		Tags:                []string{"pii", "privacy", "lora", "token-classification", "modernbert"},
	},

	// PII Detection - ModernBERT (Token-level)
	{
		LocalPath:        "models/mom-mmbert-pii-detector",
		RepoID:           "vllm-sr/mmbert-pii-detector-merged",
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
		LocalPath:           "models/mom-jailbreak-classifier",
		RepoID:              "vllm-sr/jailbreak_classifier_modernbert-base_model",
		Aliases:             []string{"jailbreak-detector", "prompt-guard", "safety-classifier", "jailbreak_classifier_modernbert-base_model", "lora_jailbreak_classifier_bert-base-uncased_model", "jailbreak_classifier_modernbert_model"},
		Purpose:             PurposeJailbreakDetection,
		Description:         "ModernBERT-based jailbreak/prompt injection detector. Model weights trained on 512-token context. Can potentially be used with ModernBERT-base-32k base model, but classifier weights were not trained for extended context.",
		ParameterSize:       "149M",
		UsesLoRA:            false,
		NumClasses:          2,     // benign/jailbreak
		MaxContextLength:    512,   // Classifier weights trained on 512 tokens - safe maximum
		BaseModelMaxContext: 32768, // Base model (ModernBERT-base-32k) supports 32K, but use with caution beyond MaxContextLength
		Tags:                []string{"safety", "jailbreak", "prompt-injection", "modernbert"},
	},

	// Vela Halu has a pair-input contract distinct from the legacy detector.
	{
		LocalPath:               "models/Vela-1.0-Encoder-307M-Halu",
		RepoID:                  "vllm-sr/Vela-1.0-Encoder-307M-Halu",
		Revision:                "ca87531211e414ac21c641b2faa8b8e21619de8f",
		DownloadExcludePatterns: velaTrainingArtifactPatterns,
		Aliases:                 []string{"Vela-1.0-Encoder-307M-Halu"},
		Purpose:                 PurposeHallucinationDetector,
		Description:             "Answer grounding against evidence and a user request with token-level hallucination spans.",
		ParameterSize:           "307M",
		EmbeddingDim:            768,
		NumClasses:              2,
		MaxContextLength:        8192,
		BaseModelMaxContext:     32768,
		DefaultAdapter:          "vela_halu",
		Tags:                    []string{"vela", "hallucination", "multilingual", "token-classification"},
	},

	// Hallucination Detection - Sentinel
	{
		LocalPath:        "models/mom-halugate-sentinel",
		RepoID:           "vllm-sr/halugate-sentinel",
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

	// Hallucination Detection - Detector (multilingual)
	{
		LocalPath:        "models/lettucedect-v2-mmbert-base",
		RepoID:           "KRLabsOrg/lettucedect-v2-mmbert-base",
		Aliases:          []string{"hallucination-detector-multilingual", "lettucedect-v2-mmbert"},
		Purpose:          PurposeHallucinationDetector,
		Description:      "Multilingual mmBERT hallucination detector covering prose, code, tool output and structured documents",
		ParameterSize:    "307M",
		EmbeddingDim:     768,
		MaxContextLength: 8192,
		Tags:             []string{"hallucination", "mmbert", "multilingual", "verification"},
	},

	// Feedback Detection
	{
		LocalPath:        "models/mom-feedback-detector",
		RepoID:           "vllm-sr/feedback-detector",
		Aliases:          []string{"feedback-detector", "user-feedback-classifier"},
		Purpose:          PurposeFeedbackDetection,
		Description:      "ModernBERT-based user feedback classifier for 4 feedback types",
		ParameterSize:    "149M",
		NumClasses:       4, // satisfied/need_clarification/wrong_answer/want_different
		MaxContextLength: 8192,
		Tags:             []string{"feedback", "classification", "modernbert", "user-intent"},
	},

	// Modality Detection - mmBERT-32K Router Classifier
	{
		LocalPath:           "models/mmbert32k-modality-router-merged",
		RepoID:              "vllm-sr/mmbert32k-modality-router-merged",
		Aliases:             []string{"modality-classifier", "modality-router", "mmbert32k-modality-router"},
		Purpose:             PurposeModalityDetection,
		Description:         "mmBERT-32K classifier for AR, DIFFUSION, and BOTH modality routing decisions",
		ParameterSize:       "307M",
		NumClasses:          3, // AR / DIFFUSION / BOTH
		MaxContextLength:    512,
		BaseModelMaxContext: 32768,
		Tags:                []string{"modality", "classification", "mmbert-32k", "multimodal", "routing"},
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
		LocalPath:        "models/mmbert-embed-32k-2d-matryoshka",
		RepoID:           "vllm-sr/mmbert-embed-32k-2d-matryoshka",
		Aliases:          []string{"mom-embedding-ultra", "mmbert-embed-32k-2d-matryoshka", "mmbert-embedding", "embedding-mmbert", "mmbert", "embedding-ultra"},
		Purpose:          PurposeEmbedding,
		Description:      "Multilingual 2D Matryoshka embedding model with 32K context, 64-768 dimension truncation, and 1800+ language coverage.",
		ParameterSize:    "307M",
		EmbeddingDim:     768, // Default, supports 512/256/128/64 via Matryoshka
		MaxContextLength: 32768,
		Tags:             []string{"embedding", "matryoshka", "2d-matryoshka", "multilingual", "modernbert", "long-context", "early-exit", "flash-attention-2"},
	},

	// Vela 1.0 Omni: the model runtime serves the pinned release from its own
	// download of the published weights and configs.
	{
		LocalPath:     "models/vela-1.0-omni-nano",
		RepoID:        "vllm-sr/Vela-1.0-Omni-Nano",
		Revision:      "2ff2d66385dbdd661a560ec3e8bcb45a0527d92e",
		Aliases:       []string{"Vela-1.0-Omni-Nano", "vela-1.0-omni-nano", "omni-nano"},
		Purpose:       PurposeEmbedding,
		Description:   "Vela Omni Nano text, image, and raw audio embeddings in one normalized 384-dimensional space.",
		ParameterSize: "164M", EmbeddingDim: 384, MaxContextLength: 512,
		DefaultAdapter:     "vela_omni",
		RuntimeProvisioned: true,
		Tags:               []string{"embedding", "multimodal", "text", "image", "audio"},
	},
	{
		LocalPath:     "models/vela-1.0-omni-mini",
		RepoID:        "vllm-sr/Vela-1.0-Omni-Mini",
		Revision:      "801bae3ad28df6891408f0e0441c676b30e132e3",
		Aliases:       []string{"Vela-1.0-Omni-Mini", "vela-1.0-omni-mini", "omni-mini"},
		Purpose:       PurposeEmbedding,
		Description:   "Vela Omni Mini text, image, and raw audio embeddings in one normalized 768-dimensional space with 32K text input.",
		ParameterSize: "1.36B", EmbeddingDim: 768, MaxContextLength: 32768,
		DefaultAdapter:     "vela_omni",
		RuntimeProvisioned: true,
		Tags:               []string{"embedding", "multimodal", "text", "image", "audio", "long-context"},
	},

	// Embedding Models - Multi-Modal (Text/Image/Audio)
	{
		LocalPath:        "models/mom-embedding-multimodal",
		RepoID:           "vllm-sr/multi-modal-embed-small",
		Aliases:          []string{"multi-modal-embed-small", "multimodal-embedding", "embedding-multimodal", "multimodal", "mom-embedding-multimodal"},
		Purpose:          PurposeEmbedding,
		Description:      "Multi-modal embedding model for text/image/audio retrieval and cross-modal matching",
		ParameterSize:    "~120M",
		EmbeddingDim:     384,
		MaxContextLength: 512,
		Tags:             []string{"embedding", "multimodal", "text", "image", "audio", "cross-modal"},
	},

	// ============================================================================
	// mmBERT-32K LoRA Models (32K context, YaRN RoPE scaling, multilingual)
	// Reference: https://huggingface.co/vllm-sr/mmbert-32k-yarn
	// ============================================================================

	// mmBERT-32K Intent Classifier
	{
		LocalPath:        "models/mmbert32k-intent-classifier-lora",
		RepoID:           "vllm-sr/mmbert32k-intent-classifier-lora",
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
		RepoID:           "vllm-sr/mmbert32k-factcheck-classifier-lora",
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
		RepoID:           "vllm-sr/mmbert32k-jailbreak-detector-lora",
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
		RepoID:           "vllm-sr/mmbert32k-feedback-detector-lora",
		Aliases:          []string{"mmbert32k-feedback", "mmbert-32k-feedback", "feedback-detector-32k"},
		Purpose:          PurposeFeedbackDetection,
		Description:      "LoRA-based 4-class user feedback classifier on top of mmbert-32k-yarn.",
		ParameterSize:    "307M + LoRA",
		UsesLoRA:         true,
		NumClasses:       4, // SAT / NEED_CLARIFICATION / WRONG_ANSWER / WANT_DIFFERENT
		MaxContextLength: 32768,
		Tags:             []string{"feedback", "classification", "lora", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Feedback Detector (Merged - for Rust/Go inference)
	{
		LocalPath:           "models/mmbert32k-feedback-detector-merged",
		RepoID:              "vllm-sr/mmbert32k-feedback-detector-merged",
		Aliases:             []string{"mmbert32k-feedback-merged", "feedback-detector-32k-merged"},
		Purpose:             PurposeFeedbackDetection,
		Description:         "Merged 4-class user feedback classifier based on mmbert-32k-yarn for direct inference without PEFT.",
		ParameterSize:       "307M",
		UsesLoRA:            false,
		NumClasses:          4, // SAT / NEED_CLARIFICATION / WRONG_ANSWER / WANT_DIFFERENT
		MaxContextLength:    512,
		BaseModelMaxContext: 32768,
		Tags:                []string{"feedback", "classification", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Intent Classifier (Merged)
	{
		LocalPath:           "models/mmbert32k-intent-classifier-merged",
		RepoID:              "vllm-sr/mmbert32k-intent-classifier-merged",
		Aliases:             []string{"mmbert32k-intent-merged", "intent-classifier-32k-merged"},
		Purpose:             PurposeDomainClassification,
		Description:         "Merged intent classifier for 14 MMLU-Pro style categories based on mmbert-32k-yarn, ready for direct inference.",
		ParameterSize:       "307M",
		UsesLoRA:            false,
		NumClasses:          14,
		MaxContextLength:    512,
		BaseModelMaxContext: 32768,
		Tags:                []string{"classification", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Fact-Check Classifier (Merged)
	{
		LocalPath:           "models/mmbert32k-factcheck-classifier-merged",
		RepoID:              "vllm-sr/mmbert32k-factcheck-classifier-merged",
		Aliases:             []string{"mmbert32k-factcheck-merged", "factcheck-classifier-32k-merged"},
		Purpose:             PurposeHallucinationSentinel,
		Description:         "Merged two-label fact-check classifier based on mmbert-32k-yarn for direct inference without PEFT.",
		ParameterSize:       "307M",
		UsesLoRA:            false,
		NumClasses:          2,
		MaxContextLength:    512,
		BaseModelMaxContext: 32768,
		Tags:                []string{"factcheck", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K Jailbreak Detector (Merged)
	{
		LocalPath:           "models/mmbert32k-jailbreak-detector-merged",
		RepoID:              "vllm-sr/mmbert32k-jailbreak-detector-merged",
		Aliases:             []string{"mmbert32k-jailbreak-merged", "jailbreak-detector-32k-merged"},
		Purpose:             PurposeJailbreakDetection,
		Description:         "Merged jailbreak and prompt-injection detector based on mmbert-32k-yarn with 32K context support.",
		ParameterSize:       "307M",
		UsesLoRA:            false,
		NumClasses:          2,
		MaxContextLength:    512,
		BaseModelMaxContext: 32768,
		Tags:                []string{"safety", "jailbreak", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K PII Detector (Merged)
	{
		LocalPath:           "models/mmbert32k-pii-detector-merged",
		RepoID:              "vllm-sr/mmbert32k-pii-detector-merged",
		Aliases:             []string{"mmbert32k-pii-merged", "pii-detector-32k-merged"},
		Purpose:             PurposePIIDetection,
		Description:         "Merged PII detector for 17 entity types and 35 BIO labels, based on mmbert-32k-yarn.",
		ParameterSize:       "307M",
		UsesLoRA:            false,
		NumClasses:          35,
		MaxContextLength:    512,
		BaseModelMaxContext: 32768,
		Tags:                []string{"pii", "privacy", "merged", "mmbert-32k", "yarn", "multilingual"},
	},

	// mmBERT-32K PII Detector
	{
		LocalPath:        "models/mmbert32k-pii-detector-lora",
		RepoID:           "vllm-sr/mmbert32k-pii-detector-lora",
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
	return findModelByPath(path)
}

func findModelByPath(path string) *ModelSpec {
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
