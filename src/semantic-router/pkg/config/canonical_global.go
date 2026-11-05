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

	// legacyModelSources, legacyFieldWarnings and authoredThresholds carry
	// what readLegacyModuleModels and authoredModuleThresholds read from
	// the document to the router configuration.
	legacyModelSources  map[string]string
	legacyFieldWarnings []ConfigWarning
	authoredThresholds  map[string]bool
	// noModel are the modules a document turns off with model_ref: "".
	noModel map[string]bool
}

// CanonicalRouterGlobal captures router-engine control knobs.
type CanonicalRouterGlobal struct {
	Enabled           *bool                    `yaml:"enabled,omitempty"`
	ConfigSource      ConfigSource             `yaml:"config_source,omitempty"`
	Strategy          RoutingStrategy          `yaml:"strategy,omitempty"`
	ListBackendModels bool                     `yaml:"list_backend_models"`
	ClearRouteCache   bool                     `yaml:"clear_route_cache"`
	StreamedBody      CanonicalStreamedBody    `yaml:"streamed_body"`
	SkipProcessing    SkipProcessingConfig     `yaml:"skip_processing"`
	ModelSelection    ModelSelectionConfig     `yaml:"model_selection"`
	Learning          RouterLearningConfig     `yaml:"learning,omitempty"`
	Fallback          *fallback.FallbackPolicy `yaml:"fallback,omitempty" json:"fallback,omitempty"`
	// AllowLossyTranslation names protocol losses translated with a warning
	// instead of rejected. Every other loss keeps failing closed.
	AllowLossyTranslation []string `yaml:"allow_lossy_translation,omitempty"`
	// RoutingInput chooses the user text that routing reads.
	RoutingInput RoutingInputConfig `yaml:"routing_input,omitempty"`
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
	KVTransfer *KVTransferConfig `yaml:"kv_transfer,omitempty"`
	Tools      ToolsConfig       `yaml:"tools"`
	Looper     LooperConfig      `yaml:"looper"`
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

// CanonicalEmbeddingModels groups embedding-related model assets. Default and
// Multimodal name the deployments that embed text and images; Semantic holds
// the embedding settings, and its model paths and endpoint are the
// deprecated way to choose those models.
type CanonicalEmbeddingModels struct {
	Default    string          `yaml:"default,omitempty"`
	Multimodal string          `yaml:"multimodal,omitempty"`
	Semantic   EmbeddingModels `yaml:"semantic"`
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
	// DecisionModel selects the shared default judgment deployment.
	DecisionModel         DecisionModelBinding `yaml:"decision_model,omitempty"`
	Safety                string               `yaml:"safety,omitempty"`
	Hazard                string               `yaml:"hazard,omitempty"`
	PromptGuard           string               `yaml:"prompt_guard,omitempty"`
	DomainClassifier      string               `yaml:"domain_classifier,omitempty"`
	PIIClassifier         string               `yaml:"pii_classifier,omitempty"`
	FactCheckClassifier   string               `yaml:"fact_check_classifier,omitempty"`
	HallucinationDetector string               `yaml:"hallucination_detector,omitempty"`
	FeedbackDetector      string               `yaml:"feedback_detector,omitempty"`
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
		if err := validateDecisionModel(&defaults); err != nil {
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
	if err := validateDecisionModel(&resolved); err != nil {
		return CanonicalGlobal{}, err
	}
	if err := readLegacyModuleModels(&resolved, rawOverride); err != nil {
		return CanonicalGlobal{}, err
	}
	resolved.authoredThresholds = authoredModuleThresholds(rawOverride)
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
	cfg.legacyModelSources = global.legacyModelSources
	cfg.noModel = global.noModel
	cfg.ModelResourceWarnings = append(cfg.ModelResourceWarnings, global.legacyFieldWarnings...)
	cfg.AuthoredThresholds = global.authoredThresholds
	return nil
}

func applyCanonicalRouterGlobal(cfg *RouterConfig, router CanonicalRouterGlobal) {
	cfg.RoutingDefaults = RoutingDefaults{Strategy: router.Strategy, Fallback: router.Fallback.Clone()}
	cfg.ConfigSource = router.ConfigSource
	cfg.Strategy = router.Strategy

	cfg.ListBackendModels = router.ListBackendModels
	cfg.RouterEnabled = router.Enabled
	cfg.ClearRouteCache = router.ClearRouteCache
	cfg.StreamedBodyMode = router.StreamedBody.Enabled
	cfg.MaxStreamedBodyBytes = router.StreamedBody.MaxBytes
	cfg.StreamedBodyTimeoutSec = router.StreamedBody.TimeoutSec
	cfg.SkipProcessing = router.SkipProcessing
	cfg.ModelSelection = router.ModelSelection
	cfg.RouterLearning = router.Learning
	cfg.Fallback = router.Fallback.Clone()
	cfg.AllowLossyTranslation = router.AllowLossyTranslation
	cfg.RoutingInput = router.RoutingInput
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
	cfg.KVTransfer = integrations.KVTransfer
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
	cfg.DecisionModel = modelCatalog.System.DecisionModel.Deployment
	cfg.EmbeddingDeployment = modelCatalog.Embeddings.Default
	cfg.MultimodalEmbeddingDeployment = modelCatalog.Embeddings.Multimodal
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

// legacyModule is one built-in module's deprecated model fields: its
// model_id and model_ref, and the system line it reads without a model_ref.
type legacyModule struct {
	consumer string
	path     string
	modelID  *string
	modelRef *string
	line     string
}

func legacyModules(global *CanonicalGlobal) []legacyModule {
	m := &global.ModelCatalog.Modules
	return []legacyModule{
		{bindingKeySafety, "global.model_catalog.modules.safety.safety", &m.Safety.Safety.ModelID, &m.Safety.Safety.ModelRef, "safety"},
		{bindingKeyHazard, "global.model_catalog.modules.safety.hazard", &m.Safety.Hazard.ModelID, &m.Safety.Hazard.ModelRef, ""},
		{"prompt_guard", "global.model_catalog.modules.prompt_guard", &m.PromptGuard.ModelID, &m.PromptGuard.ModelRef, "prompt_guard"},
		{"domain_classifier", "global.model_catalog.modules.classifier.domain", &m.Classifier.Domain.ModelID, &m.Classifier.Domain.ModelRef, "domain_classifier"},
		{"pii_classifier", "global.model_catalog.modules.classifier.pii", &m.Classifier.PII.ModelID, &m.Classifier.PII.ModelRef, "pii_classifier"},
		{"fact_check_classifier", "global.model_catalog.modules.hallucination_mitigation.fact_check", &m.HallucinationMitigation.FactCheck.ModelID, &m.HallucinationMitigation.FactCheck.ModelRef, "fact_check_classifier"},
		{"hallucination_detector", "global.model_catalog.modules.hallucination_mitigation.detector", &m.HallucinationMitigation.Detector.ModelID, &m.HallucinationMitigation.Detector.ModelRef, "hallucination_detector"},
		{"feedback_detector", "global.model_catalog.modules.feedback_detector", &m.FeedbackDetector.ModelID, &m.FeedbackDetector.ModelRef, "feedback_detector"},
	}
}

// readLegacyModuleModels resolves each module's deprecated model: its
// model_id, else the system line its model_ref names, else its own system
// line (the hazard module reads none without a model_ref). The model goes in
// the module's model_id, which model resource translation reads as a
// deployment and binding; model_ref and the system lines are cleared.
func readLegacyModuleModels(global *CanonicalGlobal, raw *StructuredPayload) error {
	system := &global.ModelCatalog.System
	lines := make(map[string]string, len(systemLines))
	for _, line := range systemLines {
		lines[line.key] = strings.TrimSpace(*line.value(system))
	}
	authored := rawModuleKeys(raw)
	global.legacyModelSources = map[string]string{}
	for _, module := range legacyModules(global) {
		ref, modelID := strings.TrimSpace(*module.modelRef), strings.TrimSpace(*module.modelID)
		*module.modelRef = ""
		source := module.path + ".model_id"
		if modelID == "" {
			line := module.line
			if ref == "" && authored[module.path+".model_ref"] {
				line = ""
			}
			if ref != "" {
				if _, known := lines[ref]; !known {
					return fmt.Errorf("%s: unknown model_ref %q", module.path, ref)
				}
				line = ref
				source = module.path + ".model_ref"
			}
			switch {
			case line != "" && lines[line] != "":
				modelID = lines[line]
				if ref == "" {
					source = "global.model_catalog.system." + line
				}
			case ref == "hazard":
				// An unset hazard line was Vela 1.0 Hazard; every other line
				// was the decision model's model, which the module follows.
				modelID = vela1HazardModel
			}
		}
		*module.modelID = modelID
		if modelID != "" {
			global.legacyModelSources[module.consumer] = source
		}
		// model_ref: "" without a model_id ran no model, which a safety
		// rule answers with the decision model instead.
		if modelID == "" && ref == "" && authored[module.path+".model_ref"] && module.line != "" && module.consumer != bindingKeySafety {
			if global.noModel == nil {
				global.noModel = map[string]bool{}
			}
			global.noModel[module.consumer] = true
			global.legacyFieldWarnings = append(global.legacyFieldWarnings, ConfigWarning{
				Code: warningDeprecatedModelField, Field: module.path + ".model_ref",
				Message: `model_ref: "" is deprecated; the module still runs no model. Set the module's enabled: false, or remove the signal`,
			})
			continue
		}
		for _, key := range []string{"model_id", "model_ref"} {
			if authored[module.path+"."+key] && modelID == "" {
				global.legacyFieldWarnings = append(global.legacyFieldWarnings, ConfigWarning{
					Code: warningDeprecatedModelField, Field: module.path + "." + key,
					Message: key + " is deprecated; an empty value has no effect: the signal follows its binding or the decision model",
				})
			}
		}
	}
	for _, line := range systemLines {
		*line.value(system) = ""
	}
	return nil
}

// rawModuleKeys lists the model_id and model_ref keys a configuration
// writes under its modules, as "<module path>.<key>".
func rawModuleKeys(raw *StructuredPayload) map[string]bool {
	keys := map[string]bool{}
	if raw == nil {
		return keys
	}
	global, err := raw.AsStringMap()
	if err != nil {
		return keys
	}
	modules := nestedStringMap(nestedStringMap(global["model_catalog"])["modules"])
	paths := map[string]map[string]interface{}{
		"global.model_catalog.modules.safety.safety":                       nestedStringMap(nestedStringMap(modules["safety"])["safety"]),
		"global.model_catalog.modules.safety.hazard":                       nestedStringMap(nestedStringMap(modules["safety"])["hazard"]),
		"global.model_catalog.modules.prompt_guard":                        nestedStringMap(modules["prompt_guard"]),
		"global.model_catalog.modules.classifier.domain":                   nestedStringMap(nestedStringMap(modules["classifier"])["domain"]),
		"global.model_catalog.modules.classifier.pii":                      nestedStringMap(nestedStringMap(modules["classifier"])["pii"]),
		"global.model_catalog.modules.hallucination_mitigation.fact_check": nestedStringMap(nestedStringMap(modules["hallucination_mitigation"])["fact_check"]),
		"global.model_catalog.modules.hallucination_mitigation.detector":   nestedStringMap(nestedStringMap(modules["hallucination_mitigation"])["detector"]),
		"global.model_catalog.modules.feedback_detector":                   nestedStringMap(modules["feedback_detector"]),
	}
	for path, module := range paths {
		for _, key := range []string{"model_id", "model_ref"} {
			if hasRawKey(module, key) {
				keys[path+"."+key] = true
			}
		}
	}
	return keys
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
