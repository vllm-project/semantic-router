package config

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// CanonicalGlobal contains router-managed runtime defaults plus sparse
// overrides, organized into explicit platform modules. Stores and services
// say where the Router keeps state; a plugin turns a feature on.
type CanonicalGlobal struct {
	Router       CanonicalRouterGlobal      `yaml:"router"`
	Services     CanonicalServiceGlobal     `yaml:"services"`
	Stores       CanonicalStoreGlobal       `yaml:"stores"`
	Integrations CanonicalIntegrationGlobal `yaml:"integrations"`
	ModelCatalog CanonicalModelCatalog      `yaml:"model_catalog"`
	// Plugins are the default plugins of every decision, and of a request that
	// matches no decision. A recipe's plugin of the same type overrides one
	// field by field, and a decision's plugin overrides the recipe's.
	Plugins []DecisionPlugin `yaml:"plugins,omitempty"`
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
	// ContextWindow governs the check of a prompt against the context
	// windows of the models that can serve it.
	ContextWindow ContextWindowConfig `yaml:"context_window,omitempty"`
	// DrainTimeoutSeconds bounds how long the standalone gateway lets the
	// requests in flight finish once the Router begins to shut down. Nil
	// keeps DefaultDrainTimeout, and 0 is no drain.
	DrainTimeoutSeconds *int `yaml:"drain_timeout_seconds,omitempty" json:"drain_timeout_seconds,omitempty" jsonschema:"minimum=0,maximum=86400"`
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

// CanonicalModelCatalog groups the Router's model assets: the deployments,
// their bindings and admission, and the embedding and decision models.
type CanonicalModelCatalog struct {
	Bindings    map[string]ModelBinding    `yaml:"bindings,omitempty"`
	Deployments map[string]ModelDeployment `yaml:"deployments,omitempty"`
	Embeddings  CanonicalEmbeddingModels   `yaml:"embeddings"`
	System      CanonicalSystemModels      `yaml:"system"`
	Admission   map[string]AdmissionConfig `yaml:"admission,omitempty"`
	// SignalTimeoutMs is the deadline of a request's model-runtime signals,
	// below the request's; 0 derives it from the request's deadline.
	SignalTimeoutMs int `yaml:"signal_timeout_ms,omitempty"`
	// OnSignalOverload is what a request gets when a rule that fails closed
	// could not be read because the decision runtime was overloaded.
	OnSignalOverload *SignalOverloadPolicy `yaml:"on_signal_overload,omitempty"`
}

// CanonicalEmbeddingModels names the embedding deployment recall uses, the
// way system.decision_model names the decision model's. Default embeds for
// every store and collection that names none of its own: the pre-declared
// vela-embedding, started only when a recall consumer needs it.
type CanonicalEmbeddingModels struct {
	Default EmbeddingModelBinding `yaml:"default,omitempty"`
}

// EmbeddingModelBinding names the deployment that embeds for recall.
type EmbeddingModelBinding struct {
	Deployment string `yaml:"deployment" json:"deployment" jsonschema:"required,minLength=1"`
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
	if err := validateDecisionModel(&resolved); err != nil {
		return CanonicalGlobal{}, err
	}
	if name := resolved.ModelCatalog.Embeddings.Default.Deployment; name == "" || name != strings.TrimSpace(name) {
		return CanonicalGlobal{}, fmt.Errorf("global.model_catalog.embeddings.default.deployment must name a declared deployment")
	}
	return resolved, nil
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

// refuseEmptyGlobalValue refuses an empty value in global or its mappings.
// Merged onto the defaults, it would replace them with zero values.
func refuseEmptyGlobalValue(global interface{}) error {
	data, err := yaml.Marshal(DefaultCanonicalGlobal())
	if err != nil {
		return err
	}
	defaults, err := parseRawConfigMap(data)
	if err != nil {
		return err
	}
	path, replaced := emptyMappingValue(global, defaults, "global")
	switch {
	case path == "":
		return nil
	case len(nestedStringMap(replaced)) > 0:
		return fmt.Errorf("%s is empty: remove it to keep its defaults, or give it fields", path)
	default:
		return fmt.Errorf("%s is empty: remove it to keep its default, or give it a value", path)
	}
}

// emptyMappingValue returns the path of the first empty value in value or its
// mappings, or "", and the default it replaces. Lists are replaced whole, so
// their items are not checked.
func emptyMappingValue(value, defaults interface{}, path string) (string, interface{}) {
	switch value.(type) {
	case nil:
		return path, defaults
	case map[string]interface{}, map[interface{}]interface{}:
		fields, defaultFields := nestedStringMap(value), nestedStringMap(defaults)
		for _, key := range slices.Sorted(maps.Keys(fields)) {
			if empty, replaced := emptyMappingValue(fields[key], defaultFields[key], path+"."+key); empty != "" {
				return empty, replaced
			}
		}
	}
	return "", nil
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
	cfg.ContextWindow = router.ContextWindow
	cfg.DrainTimeoutSeconds = router.DrainTimeoutSeconds
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
}

func applyCanonicalModelCatalogGlobal(cfg *RouterConfig, modelCatalog CanonicalModelCatalog) {
	cfg.DecisionModel = modelCatalog.System.DecisionModel.Deployment
	cfg.EmbeddingDeployment = modelCatalog.Embeddings.Default.Deployment
	cfg.ModelDeployments = cloneModelMap(modelCatalog.Deployments)
	cfg.GlobalModelBindings = cloneModelMap(modelCatalog.Bindings)
	cfg.ModelAdmission = cloneAdmissionMap(modelCatalog.Admission)
	cfg.ModelSignalTimeoutMs = modelCatalog.SignalTimeoutMs
	cfg.ModelSignalOverload = cloneSignalOverloadPolicy(modelCatalog.OnSignalOverload)
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
