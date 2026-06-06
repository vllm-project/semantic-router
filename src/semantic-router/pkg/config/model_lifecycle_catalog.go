package config

import "path"

// ModelLifecycleRole is the stable router-owned binding name for a model asset.
// User config exposes these bindings through global.model_catalog.system and
// module model_ref fields; runtime code uses the same names when building a
// download/init plan.
type ModelLifecycleRole string

const (
	ModelLifecycleRoleQwen3Embedding      ModelLifecycleRole = "qwen3_embedding"
	ModelLifecycleRoleGemmaEmbedding      ModelLifecycleRole = "gemma_embedding"
	ModelLifecycleRoleMmBERTEmbedding     ModelLifecycleRole = "mmbert_embedding"
	ModelLifecycleRoleMultiModalEmbedding ModelLifecycleRole = "multimodal_embedding"
	ModelLifecycleRoleBERTEmbedding       ModelLifecycleRole = "bert_embedding"
	ModelLifecycleRoleDomainClassifier    ModelLifecycleRole = "domain_classifier"
	ModelLifecycleRolePIIClassifier       ModelLifecycleRole = "pii_classifier"
	ModelLifecycleRolePromptGuard         ModelLifecycleRole = "prompt_guard"
	ModelLifecycleRoleFactCheckClassifier ModelLifecycleRole = "fact_check_classifier"
	ModelLifecycleRoleHallucinationModel  ModelLifecycleRole = "hallucination_detector"
	ModelLifecycleRoleHallucinationNLI    ModelLifecycleRole = "hallucination_explainer"
	ModelLifecycleRoleFeedbackDetector    ModelLifecycleRole = "feedback_detector"
	ModelLifecycleRoleModalityClassifier  ModelLifecycleRole = "modality_classifier"
)

type ModelLifecycleKind string

const (
	ModelLifecycleKindEmbedding  ModelLifecycleKind = "embedding"
	ModelLifecycleKindClassifier ModelLifecycleKind = "classifier"
	ModelLifecycleKindDetector   ModelLifecycleKind = "detector"
)

const (
	ModelLifecycleDownloadOnUse     = "download_when_required_by_config"
	ModelLifecycleInitRouterStartup = "initialize_during_router_startup"
	ModelLifecycleInitClassifier    = "initialize_during_classifier_startup"
)

const (
	defaultQwen3EmbeddingPath         = "models/mom-embedding-pro"
	defaultGemmaEmbeddingPath         = "models/mom-embedding-flash"
	defaultMmBERTEmbeddingPath        = "models/mmbert-embed-32k-2d-matryoshka"
	defaultMultiModalEmbeddingPath    = "models/mom-embedding-multimodal"
	defaultBERTEmbeddingPath          = "models/mom-embedding-light"
	defaultDomainClassifierPath       = "models/mmbert32k-intent-classifier-merged"
	defaultPIIClassifierPath          = "models/mmbert32k-pii-detector-merged"
	defaultPromptGuardPath            = "models/mmbert32k-jailbreak-detector-merged"
	defaultFactCheckClassifierPath    = "models/mmbert32k-factcheck-classifier-merged"
	defaultHallucinationDetectorPath  = "models/mom-halugate-detector"
	defaultHallucinationExplainerPath = "models/mom-halugate-explainer"
	defaultFeedbackDetectorPath       = "models/mmbert32k-feedback-detector-merged"
	defaultModalityClassifierPath     = "models/mmbert32k-modality-router-merged"
)

// ModelLifecycleBinding describes the built-in model asset bound to a stable
// runtime role. The default path must resolve through DefaultModelRegistry.
type ModelLifecycleBinding struct {
	Role                 ModelLifecycleRole
	Kind                 ModelLifecycleKind
	RuntimeName          string
	DefaultLocalPath     string
	DefaultCompanionFile string
	DownloadTiming       string
	InitializationTiming string
}

// DefaultModelLifecycleBindings returns the built-in router model bindings in
// a stable order used by config defaults, download planning, and API reporting.
func DefaultModelLifecycleBindings() []ModelLifecycleBinding {
	bindings := embeddingLifecycleBindings()
	bindings = append(bindings, classifierLifecycleBindings()...)
	bindings = append(bindings, hallucinationLifecycleBindings()...)
	bindings = append(bindings, feedbackAndModalityLifecycleBindings()...)
	return bindings
}

func embeddingLifecycleBindings() []ModelLifecycleBinding {
	return []ModelLifecycleBinding{
		{
			Role:                 ModelLifecycleRoleQwen3Embedding,
			Kind:                 ModelLifecycleKindEmbedding,
			RuntimeName:          "qwen3",
			DefaultLocalPath:     defaultQwen3EmbeddingPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitRouterStartup,
		},
		{
			Role:                 ModelLifecycleRoleGemmaEmbedding,
			Kind:                 ModelLifecycleKindEmbedding,
			RuntimeName:          "gemma",
			DefaultLocalPath:     defaultGemmaEmbeddingPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitRouterStartup,
		},
		{
			Role:                 ModelLifecycleRoleMmBERTEmbedding,
			Kind:                 ModelLifecycleKindEmbedding,
			RuntimeName:          "mmbert",
			DefaultLocalPath:     defaultMmBERTEmbeddingPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitRouterStartup,
		},
		{
			Role:                 ModelLifecycleRoleMultiModalEmbedding,
			Kind:                 ModelLifecycleKindEmbedding,
			RuntimeName:          "multimodal",
			DefaultLocalPath:     defaultMultiModalEmbeddingPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitRouterStartup,
		},
		{
			Role:                 ModelLifecycleRoleBERTEmbedding,
			Kind:                 ModelLifecycleKindEmbedding,
			RuntimeName:          "bert",
			DefaultLocalPath:     defaultBERTEmbeddingPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitRouterStartup,
		},
	}
}

func classifierLifecycleBindings() []ModelLifecycleBinding {
	return []ModelLifecycleBinding{
		{
			Role:                 ModelLifecycleRoleDomainClassifier,
			Kind:                 ModelLifecycleKindClassifier,
			RuntimeName:          "domain_classifier",
			DefaultLocalPath:     defaultDomainClassifierPath,
			DefaultCompanionFile: "category_mapping.json",
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitClassifier,
		},
		{
			Role:                 ModelLifecycleRolePIIClassifier,
			Kind:                 ModelLifecycleKindClassifier,
			RuntimeName:          "pii_classifier",
			DefaultLocalPath:     defaultPIIClassifierPath,
			DefaultCompanionFile: "pii_type_mapping.json",
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitClassifier,
		},
		{
			Role:                 ModelLifecycleRolePromptGuard,
			Kind:                 ModelLifecycleKindClassifier,
			RuntimeName:          "prompt_guard",
			DefaultLocalPath:     defaultPromptGuardPath,
			DefaultCompanionFile: "jailbreak_type_mapping.json",
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitClassifier,
		},
		{
			Role:                 ModelLifecycleRoleFactCheckClassifier,
			Kind:                 ModelLifecycleKindClassifier,
			RuntimeName:          "fact_check_classifier",
			DefaultLocalPath:     defaultFactCheckClassifierPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitClassifier,
		},
	}
}

func hallucinationLifecycleBindings() []ModelLifecycleBinding {
	return []ModelLifecycleBinding{
		{
			Role:                 ModelLifecycleRoleHallucinationModel,
			Kind:                 ModelLifecycleKindDetector,
			RuntimeName:          "hallucination_detector",
			DefaultLocalPath:     defaultHallucinationDetectorPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitClassifier,
		},
		{
			Role:                 ModelLifecycleRoleHallucinationNLI,
			Kind:                 ModelLifecycleKindDetector,
			RuntimeName:          "hallucination_explainer",
			DefaultLocalPath:     defaultHallucinationExplainerPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitClassifier,
		},
	}
}

func feedbackAndModalityLifecycleBindings() []ModelLifecycleBinding {
	return []ModelLifecycleBinding{
		{
			Role:                 ModelLifecycleRoleFeedbackDetector,
			Kind:                 ModelLifecycleKindClassifier,
			RuntimeName:          "feedback_detector",
			DefaultLocalPath:     defaultFeedbackDetectorPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitClassifier,
		},
		{
			Role:                 ModelLifecycleRoleModalityClassifier,
			Kind:                 ModelLifecycleKindClassifier,
			RuntimeName:          "modality_classifier",
			DefaultLocalPath:     defaultModalityClassifierPath,
			DownloadTiming:       ModelLifecycleDownloadOnUse,
			InitializationTiming: ModelLifecycleInitRouterStartup,
		},
	}
}

func ModelLifecycleBindingForRole(role ModelLifecycleRole) (ModelLifecycleBinding, bool) {
	for _, binding := range DefaultModelLifecycleBindings() {
		if binding.Role == role {
			return binding, true
		}
	}
	return ModelLifecycleBinding{}, false
}

func DefaultModelPathForLifecycleRole(role ModelLifecycleRole) string {
	binding, ok := ModelLifecycleBindingForRole(role)
	if !ok {
		return ""
	}
	return binding.DefaultLocalPath
}

func DefaultCompanionPathForLifecycleRole(role ModelLifecycleRole) string {
	binding, ok := ModelLifecycleBindingForRole(role)
	if !ok || binding.DefaultCompanionFile == "" || binding.DefaultLocalPath == "" {
		return ""
	}
	return path.Join(binding.DefaultLocalPath, binding.DefaultCompanionFile)
}
