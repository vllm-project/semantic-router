package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v2"
)

// CanonicalGlobal contains router-managed runtime defaults plus sparse
// overrides, organized into explicit platform modules.
type CanonicalGlobal struct {
	Router       CanonicalRouterGlobal      `yaml:"router"`
	Services     CanonicalServiceGlobal     `yaml:"services"`
	Stores       CanonicalStoreGlobal       `yaml:"stores"`
	Integrations CanonicalIntegrationGlobal `yaml:"integrations"`
	ModelCatalog CanonicalModelCatalog      `yaml:"model_catalog"`
}

// CanonicalRouterGlobal captures router-engine control knobs.
type CanonicalRouterGlobal struct {
	ConfigSource              ConfigSource          `yaml:"config_source,omitempty"`
	Strategy                  string                `yaml:"strategy,omitempty"`
	AutoModelName             string                `yaml:"auto_model_name,omitempty"`
	IncludeConfigModelsInList bool                  `yaml:"include_config_models_in_list"`
	ClearRouteCache           bool                  `yaml:"clear_route_cache"`
	StreamedBody              CanonicalStreamedBody `yaml:"streamed_body"`
	ModelSelection            ModelSelectionConfig  `yaml:"model_selection"`
}

// CanonicalStreamedBody groups streaming request body controls.
type CanonicalStreamedBody struct {
	Enabled    bool  `yaml:"enabled"`
	MaxBytes   int64 `yaml:"max_bytes,omitempty"`
	TimeoutSec int   `yaml:"timeout_sec,omitempty"`
}

// CanonicalServiceGlobal groups shared runtime services exposed by the router.
type CanonicalServiceGlobal struct {
	API           APIConfig           `yaml:"api"`
	ResponseAPI   ResponseAPIConfig   `yaml:"response_api"`
	Observability ObservabilityConfig `yaml:"observability"`
	Authz         AuthzConfig         `yaml:"authz"`
	RateLimit     RateLimitConfig     `yaml:"ratelimit"`
	RouterReplay  RouterReplayConfig  `yaml:"router_replay"`
	StartupStatus StartupStatusConfig `yaml:"startup_status"`
}

// CanonicalStoreGlobal groups storage-backed runtime facilities.
type CanonicalStoreGlobal struct {
	SemanticCache SemanticCache      `yaml:"semantic_cache"`
	Memory        MemoryConfig       `yaml:"memory"`
	VectorStore   *VectorStoreConfig `yaml:"vector_store,omitempty"`
}

// CanonicalIntegrationGlobal groups external helper services used by the router.
type CanonicalIntegrationGlobal struct {
	Tools  ToolsConfig  `yaml:"tools"`
	Looper LooperConfig `yaml:"looper"`
}

// CanonicalModelCatalog groups router-owned model assets and the module
// configs that resolve through those assets.
type CanonicalModelCatalog struct {
	Embeddings CanonicalEmbeddingModels `yaml:"embeddings"`
	System     CanonicalSystemModels    `yaml:"system"`
	External   []ExternalModelConfig    `yaml:"external,omitempty"`
	KBs        []KnowledgeBaseConfig    `yaml:"kbs,omitempty"`
	Modules    CanonicalModelModules    `yaml:"modules"`
}

// CanonicalEmbeddingModels groups embedding-related model assets.
type CanonicalEmbeddingModels struct {
	Semantic EmbeddingModels `yaml:"semantic"`
}

// CanonicalModelModules groups configurable capability modules built on top of
// router-owned model assets.
type CanonicalModelModules struct {
	PromptCompression       PromptCompressionConfig         `yaml:"prompt_compression"`
	PromptGuard             CanonicalPromptGuardModule      `yaml:"prompt_guard"`
	Classifier              CanonicalClassifierModule       `yaml:"classifier"`
	Complexity              ComplexityModelConfig           `yaml:"complexity"`
	HallucinationMitigation CanonicalHallucinationModule    `yaml:"hallucination_mitigation"`
	FeedbackDetector        CanonicalFeedbackDetectorModule `yaml:"feedback_detector"`
	ModalityDetector        ModalityDetectorConfig          `yaml:"modality_detector"`
}

// CanonicalSystemModels centralizes stable capability bindings for built-in models.
type CanonicalSystemModels struct {
	PromptGuard            string `yaml:"prompt_guard,omitempty"`
	DomainClassifier       string `yaml:"domain_classifier,omitempty"`
	PIIClassifier          string `yaml:"pii_classifier,omitempty"`
	FactCheckClassifier    string `yaml:"fact_check_classifier,omitempty"`
	HallucinationDetector  string `yaml:"hallucination_detector,omitempty"`
	HallucinationExplainer string `yaml:"hallucination_explainer,omitempty"`
	FeedbackDetector       string `yaml:"feedback_detector,omitempty"`
}

// CanonicalPromptGuardModule keeps prompt-guard settings visible as a module
// while resolving the concrete model from the shared system-model catalog.
type CanonicalPromptGuardModule struct {
	PromptGuardConfig `yaml:",inline"`
	ModelRef          string `yaml:"model_ref,omitempty"`
}

// CanonicalClassifierModule exposes classifier submodules explicitly.
type CanonicalClassifierModule struct {
	Domain     CanonicalCategoryModule `yaml:"domain"`
	MCP        MCPCategoryModel        `yaml:"mcp"`
	PII        CanonicalPIIModule      `yaml:"pii"`
	Preference PreferenceModelConfig   `yaml:"preference"`
}

type CanonicalCategoryModule struct {
	CategoryModel `yaml:",inline"`
	ModelRef      string `yaml:"model_ref,omitempty"`
}

type CanonicalPIIModule struct {
	PIIModel `yaml:",inline"`
	ModelRef string `yaml:"model_ref,omitempty"`
}

// CanonicalHallucinationModule keeps the mitigation block readable by splitting
// fact-check, detector, and explainer responsibilities.
type CanonicalHallucinationModule struct {
	Enabled                 bool                           `yaml:"enabled,omitempty"`
	OnHallucinationDetected string                         `yaml:"on_hallucination_detected,omitempty"`
	FactCheck               CanonicalFactCheckModule       `yaml:"fact_check"`
	Detector                CanonicalHallucinationDetector `yaml:"detector"`
	Explainer               CanonicalExplainerModule       `yaml:"explainer"`
}

type CanonicalFactCheckModule struct {
	FactCheckModelConfig `yaml:",inline"`
	ModelRef             string `yaml:"model_ref,omitempty"`
}

type CanonicalHallucinationDetector struct {
	HallucinationModelConfig `yaml:",inline"`
	ModelRef                 string `yaml:"model_ref,omitempty"`
}

type CanonicalExplainerModule struct {
	NLIModelConfig `yaml:",inline"`
	ModelRef       string `yaml:"model_ref,omitempty"`
}

type CanonicalFeedbackDetectorModule struct {
	FeedbackDetectorConfig `yaml:",inline"`
	ModelRef               string `yaml:"model_ref,omitempty"`
}

func (m CanonicalClassifierModule) runtimeConfig() Classifier {
	return Classifier{
		CategoryModel:    m.Domain.CategoryModel,
		MCPCategoryModel: m.MCP,
		PIIModel:         m.PII.PIIModel,
		PreferenceModel:  m.Preference.WithDefaults(),
	}
}

func (m CanonicalHallucinationModule) runtimeConfig() HallucinationMitigationConfig {
	return HallucinationMitigationConfig{
		Enabled:                 m.Enabled,
		FactCheckModel:          m.FactCheck.FactCheckModelConfig,
		HallucinationModel:      m.Detector.HallucinationModelConfig,
		NLIModel:                m.Explainer.NLIModelConfig,
		OnHallucinationDetected: m.OnHallucinationDetected,
	}
}

func resolveCanonicalGlobal(override *CanonicalGlobal, rawOverride *StructuredPayload) (CanonicalGlobal, error) {
	defaults := DefaultCanonicalGlobal()
	if rawOverride == nil && override == nil {
		if err := resolveModuleModelRefs(&defaults); err != nil {
			return CanonicalGlobal{}, err
		}
		return defaults, nil
	}

	resolved := defaults
	overrideSource := interface{}(override)
	if rawOverride != nil {
		overrideSource = rawOverride
	}

	overrideBytes, err := yaml.Marshal(overrideSource)
	if err != nil {
		return CanonicalGlobal{}, fmt.Errorf("failed to marshal global override: %w", err)
	}
	if err := yaml.Unmarshal(overrideBytes, &resolved); err != nil {
		return CanonicalGlobal{}, fmt.Errorf("failed to merge global override: %w", err)
	}
	if err := resolveModuleModelRefs(&resolved); err != nil {
		return CanonicalGlobal{}, err
	}
	return resolved, nil
}

func applyCanonicalGlobal(cfg *RouterConfig, global *CanonicalGlobal) error {
	if global == nil {
		return nil
	}

	cfg.ConfigSource = global.Router.ConfigSource
	cfg.Strategy = global.Router.Strategy
	cfg.AutoModelName = global.Router.AutoModelName
	cfg.IncludeConfigModelsInList = global.Router.IncludeConfigModelsInList
	cfg.ClearRouteCache = global.Router.ClearRouteCache
	cfg.StreamedBodyMode = global.Router.StreamedBody.Enabled
	cfg.MaxStreamedBodyBytes = global.Router.StreamedBody.MaxBytes
	cfg.StreamedBodyTimeoutSec = global.Router.StreamedBody.TimeoutSec
	cfg.ModelSelection = global.Router.ModelSelection

	cfg.API = global.Services.API
	cfg.ResponseAPI = global.Services.ResponseAPI
	cfg.Observability = global.Services.Observability
	cfg.Authz = global.Services.Authz
	cfg.RateLimit = global.Services.RateLimit
	cfg.RouterReplay = global.Services.RouterReplay
	cfg.StartupStatus = global.Services.StartupStatus

	cfg.SemanticCache = global.Stores.SemanticCache
	cfg.Memory = global.Stores.Memory
	cfg.VectorStore = global.Stores.VectorStore

	cfg.Tools = global.Integrations.Tools
	cfg.Looper = global.Integrations.Looper

	cfg.ExternalModels = append([]ExternalModelConfig(nil), global.ModelCatalog.External...)
	cfg.EmbeddingModels = global.ModelCatalog.Embeddings.Semantic
	cfg.KnowledgeBases = append([]KnowledgeBaseConfig(nil), global.ModelCatalog.KBs...)

	cfg.PromptCompression = global.ModelCatalog.Modules.PromptCompression
	cfg.PromptGuard = global.ModelCatalog.Modules.PromptGuard.PromptGuardConfig
	cfg.Classifier = global.ModelCatalog.Modules.Classifier.runtimeConfig()
	cfg.ComplexityModel = global.ModelCatalog.Modules.Complexity.WithDefaults()
	cfg.HallucinationMitigation = global.ModelCatalog.Modules.HallucinationMitigation.runtimeConfig()
	cfg.FeedbackDetector = global.ModelCatalog.Modules.FeedbackDetector.FeedbackDetectorConfig
	cfg.ModalityDetector = global.ModelCatalog.Modules.ModalityDetector

	return nil
}

func resolveModuleModelRefs(global *CanonicalGlobal) error {
	if global == nil {
		return nil
	}

	var err error
	if global.ModelCatalog.Modules.PromptGuard.ModelID, err = resolveSystemModelRef(
		global.ModelCatalog.Modules.PromptGuard.ModelRef,
		global.ModelCatalog.Modules.PromptGuard.ModelID,
		global.ModelCatalog.System,
	); err != nil {
		return fmt.Errorf("global.model_catalog.modules.prompt_guard: %w", err)
	}
	if global.ModelCatalog.Modules.Classifier.Domain.ModelID, err = resolveSystemModelRef(
		global.ModelCatalog.Modules.Classifier.Domain.ModelRef,
		global.ModelCatalog.Modules.Classifier.Domain.ModelID,
		global.ModelCatalog.System,
	); err != nil {
		return fmt.Errorf("global.model_catalog.modules.classifier.domain: %w", err)
	}
	if global.ModelCatalog.Modules.Classifier.PII.ModelID, err = resolveSystemModelRef(
		global.ModelCatalog.Modules.Classifier.PII.ModelRef,
		global.ModelCatalog.Modules.Classifier.PII.ModelID,
		global.ModelCatalog.System,
	); err != nil {
		return fmt.Errorf("global.model_catalog.modules.classifier.pii: %w", err)
	}
	if global.ModelCatalog.Modules.HallucinationMitigation.FactCheck.ModelID, err = resolveSystemModelRef(
		global.ModelCatalog.Modules.HallucinationMitigation.FactCheck.ModelRef,
		global.ModelCatalog.Modules.HallucinationMitigation.FactCheck.ModelID,
		global.ModelCatalog.System,
	); err != nil {
		return fmt.Errorf("global.model_catalog.modules.hallucination_mitigation.fact_check: %w", err)
	}
	if global.ModelCatalog.Modules.HallucinationMitigation.Detector.ModelID, err = resolveSystemModelRef(
		global.ModelCatalog.Modules.HallucinationMitigation.Detector.ModelRef,
		global.ModelCatalog.Modules.HallucinationMitigation.Detector.ModelID,
		global.ModelCatalog.System,
	); err != nil {
		return fmt.Errorf("global.model_catalog.modules.hallucination_mitigation.detector: %w", err)
	}
	if global.ModelCatalog.Modules.HallucinationMitigation.Explainer.ModelID, err = resolveSystemModelRef(
		global.ModelCatalog.Modules.HallucinationMitigation.Explainer.ModelRef,
		global.ModelCatalog.Modules.HallucinationMitigation.Explainer.ModelID,
		global.ModelCatalog.System,
	); err != nil {
		return fmt.Errorf("global.model_catalog.modules.hallucination_mitigation.explainer: %w", err)
	}
	if global.ModelCatalog.Modules.FeedbackDetector.ModelID, err = resolveSystemModelRef(
		global.ModelCatalog.Modules.FeedbackDetector.ModelRef,
		global.ModelCatalog.Modules.FeedbackDetector.ModelID,
		global.ModelCatalog.System,
	); err != nil {
		return fmt.Errorf("global.model_catalog.modules.feedback_detector: %w", err)
	}
	return nil
}

func resolveSystemModelRef(ref string, explicitModelID string, catalog CanonicalSystemModels) (string, error) {
	if explicitModelID != "" {
		return explicitModelID, nil
	}
	if ref == "" {
		return "", nil
	}

	var modelID string
	switch ref {
	case "prompt_guard":
		modelID = catalog.PromptGuard
	case "domain_classifier":
		modelID = catalog.DomainClassifier
	case "pii_classifier":
		modelID = catalog.PIIClassifier
	case "fact_check_classifier":
		modelID = catalog.FactCheckClassifier
	case "hallucination_detector":
		modelID = catalog.HallucinationDetector
	case "hallucination_explainer":
		modelID = catalog.HallucinationExplainer
	case "feedback_detector":
		modelID = catalog.FeedbackDetector
	default:
		return "", fmt.Errorf("unknown model_ref %q", ref)
	}
	if modelID == "" {
		return "", fmt.Errorf("model_ref %q is not configured in global.model_catalog.system", ref)
	}
	return modelID, nil
}

func resolveBackendAPIKey(ref CanonicalBackendRef) string {
	if ref.APIKey != "" {
		return ref.APIKey
	}
	if ref.APIKeyEnv != "" {
		return os.Getenv(ref.APIKeyEnv)
	}
	return ""
}
