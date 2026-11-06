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
	// Headers governs the optional response headers, such as who may turn on
	// the debug headers.
	Headers RouterHeadersConfig `yaml:"headers,omitempty"`
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

// CanonicalModelModules holds the prompt compression of the request text
// signals read.
type CanonicalModelModules struct {
	PromptCompression PromptCompressionConfig `yaml:"prompt_compression"`
}

// CanonicalSystemModels centralizes stable capability bindings for built-in models.
type CanonicalSystemModels struct {
	// DecisionModel selects the shared default judgment deployment.
	DecisionModel DecisionModelBinding `yaml:"decision_model,omitempty"`
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
	if err := validateDecisionModel(&resolved); err != nil {
		return CanonicalGlobal{}, err
	}
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
	cfg.Headers = router.Headers
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
	cfg.PromptCompression = modelCatalog.Modules.PromptCompression
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
