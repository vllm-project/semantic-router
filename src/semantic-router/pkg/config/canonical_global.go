package config

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	ConfigSource              ConfigSource             `yaml:"config_source,omitempty"`
	Strategy                  RoutingStrategy          `yaml:"strategy,omitempty"`
	AutoModelName             string                   `yaml:"auto_model_name,omitempty"`
	AutoModelNames            *[]string                `yaml:"auto_model_names,omitempty"`
	IncludeConfigModelsInList bool                     `yaml:"include_config_models_in_list"`
	ClearRouteCache           bool                     `yaml:"clear_route_cache"`
	StreamedBody              CanonicalStreamedBody    `yaml:"streamed_body"`
	SkipProcessing            SkipProcessingConfig     `yaml:"skip_processing"`
	ModelSelection            ModelSelectionConfig     `yaml:"model_selection"`
	Learning                  RouterLearningConfig     `yaml:"learning,omitempty"`
	Fallback                  *fallback.FallbackPolicy `yaml:"fallback,omitempty" json:"fallback,omitempty"`
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
	ManagementAPI ManagementAPIConfig `yaml:"management_api"`
	RouterReplay  RouterReplayConfig  `yaml:"router_replay"`
	StartupStatus StartupStatusConfig `yaml:"startup_status"`
}

// CanonicalStoreGlobal groups storage-backed runtime facilities.
type CanonicalStoreGlobal struct {
	ResponseCache ResponseCacheStoreConfig `yaml:"response_cache"`
	Memory        MemoryConfig             `yaml:"memory"`
	VectorStore   *VectorStoreConfig       `yaml:"vector_store,omitempty"`

	// ToolSessions configures session-scoped sticky tool-set selection's
	// shared store (issue #3347). Optional and pointer-shaped like
	// VectorStore: most deployments won't enable sticky selection at all.
	ToolSessions *ToolSessionStoreConfig `yaml:"tool_sessions,omitempty"`
}

// CanonicalIntegrationGlobal groups external helper services used by the router.
type CanonicalIntegrationGlobal struct {
	Tools  ToolsConfig  `yaml:"tools"`
	Looper LooperConfig `yaml:"looper"`
}

// CanonicalModelCatalog groups router-owned model assets and the module
// configs that resolve through those assets.
type CanonicalModelCatalog struct {
	Bindings    map[string]ModelBinding    `yaml:"bindings,omitempty"`
	Deployments map[string]ModelDeployment `yaml:"deployments,omitempty"`
	Embeddings  CanonicalEmbeddingModels   `yaml:"embeddings"`
	System      CanonicalSystemModels      `yaml:"system"`
	External    []ExternalModelConfig      `yaml:"external,omitempty"`
	KBs         []KnowledgeBaseConfig      `yaml:"kbs,omitempty"`
	Modules     CanonicalModelModules      `yaml:"modules"`
	Admission   map[string]AdmissionConfig `yaml:"admission,omitempty"`
	// SignalTimeoutMs is the deadline of a request's model-runtime signals,
	// below the request's; 0 derives it from the request's deadline.
	SignalTimeoutMs int `yaml:"signal_timeout_ms,omitempty"`
}

// CanonicalEmbeddingModels groups embedding-related model assets.
type CanonicalEmbeddingModels struct {
	Semantic EmbeddingModels `yaml:"semantic"`
}

// CanonicalModelModules groups configurable capability modules built on top of
// router-owned model assets.
type CanonicalModelModules struct {
	Safety                  SafetyModelsConfig              `yaml:"safety"`
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
	// DecisionModel is the Vela model that answers the built-in signals and
	// the decision questions that name no deployment (decision_model.go).
	DecisionModel         string `yaml:"decision_model,omitempty"`
	Safety                string `yaml:"safety,omitempty"`
	Hazard                string `yaml:"hazard,omitempty"`
	PromptGuard           string `yaml:"prompt_guard,omitempty"`
	DomainClassifier      string `yaml:"domain_classifier,omitempty"`
	PIIClassifier         string `yaml:"pii_classifier,omitempty"`
	FactCheckClassifier   string `yaml:"fact_check_classifier,omitempty"`
	HallucinationDetector string `yaml:"hallucination_detector,omitempty"`
	FeedbackDetector      string `yaml:"feedback_detector,omitempty"`
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
// fact-check and detector responsibilities.
type CanonicalHallucinationModule struct {
	Enabled   bool                           `yaml:"enabled,omitempty"`
	FactCheck CanonicalFactCheckModule       `yaml:"fact_check"`
	Detector  CanonicalHallucinationDetector `yaml:"detector"`
}

type CanonicalFactCheckModule struct {
	FactCheckModelConfig `yaml:",inline"`
	ModelRef             string `yaml:"model_ref,omitempty"`
}

type CanonicalHallucinationDetector struct {
	HallucinationModelConfig `yaml:",inline"`
	ModelRef                 string `yaml:"model_ref,omitempty"`
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
		Enabled:            m.Enabled,
		FactCheckModel:     m.FactCheck.FactCheckModelConfig,
		HallucinationModel: m.Detector.HallucinationModelConfig,
	}
}

func resolveCanonicalGlobal(override *CanonicalGlobal, rawOverride *StructuredPayload) (CanonicalGlobal, error) {
	defaults := DefaultCanonicalGlobal()
	if rawOverride == nil && override == nil {
		if err := applyDecisionModel(&defaults, nil); err != nil {
			return CanonicalGlobal{}, err
		}
		if err := resolveModuleModelRefs(&defaults); err != nil {
			return CanonicalGlobal{}, err
		}
		return defaults, nil
	}

	resolved, err := mergeCanonicalGlobalOverride(defaults, override, rawOverride)
	if err != nil {
		return CanonicalGlobal{}, err
	}
	normalizeSparseCanonicalEmbeddingOverride(&resolved, rawOverride)
	if err := rejectLegacyPromptGuardProtocol(rawOverride); err != nil {
		return CanonicalGlobal{}, err
	}
	if err := applyDecisionModel(&resolved, rawOverride); err != nil {
		return CanonicalGlobal{}, err
	}
	if err := resolveModuleModelRefs(&resolved); err != nil {
		return CanonicalGlobal{}, err
	}
	normalizeModuleOperatingPoints(&resolved, rawOverride)
	return resolved, nil
}

// Representation defaults belong to the default mmBERT model. Selecting a
// different family must not silently request mmBERT's layer 22 from that owner.
func normalizeSparseCanonicalEmbeddingOverride(resolved *CanonicalGlobal, raw *StructuredPayload) {
	if raw == nil {
		return
	}
	global, err := raw.AsStringMap()
	if err != nil {
		return
	}
	catalog := nestedStringMap(global["model_catalog"])
	embeddings := nestedStringMap(catalog["embeddings"])
	semantic := nestedStringMap(embeddings["semantic"])
	settings := nestedStringMap(semantic["embedding_config"])
	if model, ok := settings["model_type"].(string); ok && strings.TrimSpace(model) != "" && strings.ToLower(strings.TrimSpace(model)) != "mmbert" {
		if !hasRawKey(settings, "target_layer") {
			resolved.ModelCatalog.Embeddings.Semantic.EmbeddingConfig.TargetLayer = 0
		}
	}
}

func mergeCanonicalGlobalOverride(
	defaults CanonicalGlobal,
	override *CanonicalGlobal,
	rawOverride *StructuredPayload,
) (CanonicalGlobal, error) {
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
	return resolved, nil
}

func hasRawKey(raw map[string]interface{}, key string) bool {
	_, ok := raw[key]
	return ok
}

func applyCanonicalGlobal(cfg *RouterConfig, global *CanonicalGlobal) error {
	if global == nil {
		return nil
	}
	applyCanonicalRouterGlobal(cfg, global.Router)
	applyCanonicalServiceGlobal(cfg, global.Services)
	applyCanonicalStoreGlobal(cfg, global.Stores)
	applyCanonicalIntegrationGlobal(cfg, global.Integrations)
	applyCanonicalModelCatalogGlobal(cfg, global.ModelCatalog)
	return nil
}

func applyCanonicalRouterGlobal(cfg *RouterConfig, router CanonicalRouterGlobal) {
	cfg.ConfigSource = router.ConfigSource
	cfg.Strategy = router.Strategy
	cfg.AutoModelName = router.AutoModelName
	cfg.AutoModelNames = nil
	if router.AutoModelNames != nil {
		cfg.AutoModelNames = append([]string{}, (*router.AutoModelNames)...)
	}
	cfg.IncludeConfigModelsInList = router.IncludeConfigModelsInList
	cfg.ClearRouteCache = router.ClearRouteCache
	cfg.StreamedBodyMode = router.StreamedBody.Enabled
	cfg.MaxStreamedBodyBytes = router.StreamedBody.MaxBytes
	cfg.StreamedBodyTimeoutSec = router.StreamedBody.TimeoutSec
	cfg.SkipProcessing = router.SkipProcessing
	cfg.ModelSelection = router.ModelSelection
	cfg.RouterLearning = router.Learning
	if router.Fallback != nil && cfg.Fallback == nil {
		cfg.Fallback = router.Fallback.Clone()
	}
}

func applyCanonicalServiceGlobal(cfg *RouterConfig, services CanonicalServiceGlobal) {
	cfg.API = services.API
	cfg.ResponseAPI = services.ResponseAPI
	cfg.Observability = services.Observability
	cfg.Authz = services.Authz
	cfg.RateLimit = services.RateLimit
	cfg.ManagementAPI = services.ManagementAPI
	cfg.RouterReplay = services.RouterReplay
	cfg.StartupStatus = services.StartupStatus
}

func applyCanonicalStoreGlobal(cfg *RouterConfig, stores CanonicalStoreGlobal) {
	cfg.SemanticCache = stores.ResponseCache
	cfg.Memory = stores.Memory
	cfg.VectorStore = stores.VectorStore
	cfg.ToolSessions = stores.ToolSessions
}

func applyCanonicalIntegrationGlobal(cfg *RouterConfig, integrations CanonicalIntegrationGlobal) {
	cfg.Tools = integrations.Tools
	cfg.Looper = integrations.Looper
	if integrations.Looper.Endpoint != "" {
		logging.ComponentWarnEvent("config", "looper_endpoint_deprecated", map[string]interface{}{
			"field":  "global.integrations.looper.endpoint",
			"reason": "the Router makes Looper calls in process; the field is ignored and goes in the next release",
			"fix":    "remove it, or run vllm-sr config migrate",
		})
	}
}

func applyCanonicalModelCatalogGlobal(cfg *RouterConfig, modelCatalog CanonicalModelCatalog) {
	cfg.DecisionModel = modelCatalog.System.DecisionModel
	cfg.ModelDeployments = cloneModelMap(modelCatalog.Deployments)
	cfg.GlobalModelBindings = cloneModelMap(modelCatalog.Bindings)
	cfg.ExternalModels = append([]ExternalModelConfig(nil), modelCatalog.External...)
	cfg.EmbeddingModels = modelCatalog.Embeddings.Semantic
	cfg.KnowledgeBases = append([]KnowledgeBaseConfig(nil), modelCatalog.KBs...)
	cfg.PromptCompression = modelCatalog.Modules.PromptCompression
	cfg.PromptGuard = modelCatalog.Modules.PromptGuard.PromptGuardConfig
	cfg.Classifier = modelCatalog.Modules.Classifier.runtimeConfig()
	cfg.ComplexityModel = modelCatalog.Modules.Complexity.WithDefaults()
	cfg.HallucinationMitigation = modelCatalog.Modules.HallucinationMitigation.runtimeConfig()
	cfg.FeedbackDetector = modelCatalog.Modules.FeedbackDetector.FeedbackDetectorConfig
	cfg.ModalityDetector = modelCatalog.Modules.ModalityDetector
	cfg.SafetyModels = modelCatalog.Modules.Safety
	cfg.ModelAdmission = cloneAdmissionMap(modelCatalog.Admission)
	cfg.ModelSignalTimeoutMs = modelCatalog.SignalTimeoutMs
}

func cloneAdmissionMap(admission map[string]AdmissionConfig) map[string]AdmissionConfig {
	if len(admission) == 0 {
		return nil
	}
	cloned := make(map[string]AdmissionConfig, len(admission))
	for key, value := range admission {
		cloned[key] = value
	}
	return cloned
}

func resolveModuleModelRefs(global *CanonicalGlobal) error {
	if global == nil {
		return nil
	}

	var err error
	for name, head := range map[string]*SequenceHeadModelConfig{
		"safety": &global.ModelCatalog.Modules.Safety.Safety,
		"hazard": &global.ModelCatalog.Modules.Safety.Hazard,
	} {
		if head.ModelID, err = resolveSystemModelRef(head.ModelRef, head.ModelID, global.ModelCatalog.System); err != nil {
			return fmt.Errorf("global.model_catalog.modules.safety.%s: %w", name, err)
		}
	}
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
	case "safety":
		modelID = catalog.Safety
	case "hazard":
		modelID = catalog.Hazard
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

// rejectLegacyPromptGuardProtocol refuses the legacy protocol key, which the
// migration command converts to a named backend.
func rejectLegacyPromptGuardProtocol(rawOverride *StructuredPayload) error {
	if rawOverride == nil || rawOverride.IsEmpty() {
		return nil
	}
	var global map[string]interface{}
	if err := rawOverride.DecodeInto(&global); err != nil {
		return err
	}
	catalog := nestedStringMap(global["model_catalog"])
	modules := nestedStringMap(catalog["modules"])
	raw := nestedStringMap(modules["prompt_guard"])
	if protocol, ok := raw["protocol"].(string); ok && strings.TrimSpace(protocol) != "" {
		return fmt.Errorf("global.model_catalog.modules.prompt_guard.protocol is legacy; run vllm-sr config migrate to declare a named backend")
	}
	return nil
}
