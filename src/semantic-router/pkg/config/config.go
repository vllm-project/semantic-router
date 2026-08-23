package config

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"

// Model role constants for external models.
const (
	ModelRoleGuardrail        = "guardrail"
	ModelRoleClassification   = "classification"
	ModelRoleScoring          = "scoring"
	ModelRolePreference       = "preference"
	ModelRoleMemoryRewrite    = "memory_rewrite"
	ModelRoleMemoryExtraction = "memory_extraction"
)

// PromptGuardConfig.Variant values, selecting which local Candle-backed
// jailbreak classifier variant to use. Mutually exclusive with Protocol - see
// PromptGuardConfig's doc comment. An empty/unset value passed directly to
// createJailbreakInference falls back to PromptGuardVariantCandle. This is
// NOT the same as the canonical-config default: canonical resolution starts
// from defaultPromptGuardModule()'s baseline (PromptGuardVariantMmBERT32K,
// matching the bundled mmbert32k model it also defaults ModelID to) and
// overlays user YAML, so a canonical-resolved config with no explicit
// variant gets mmbert32k, not candle. A user who wants the plain candle
// variant under canonical resolution must set variant: candle explicitly.
const (
	// PromptGuardVariantCandle runs the bundled Candle model locally
	// (LoRA/BERT auto-detect, falling back to ModernBERT).
	PromptGuardVariantCandle = "candle"
	// PromptGuardVariantMmBERT32K runs the bundled mmBERT-32K model locally
	// (32K context, YaRN RoPE, multilingual).
	PromptGuardVariantMmBERT32K = "mmbert32k"
)

// PromptGuardConfig.Protocol values, selecting which remote HTTP wire
// contract to use for an external model with role="guardrail". Mutually
// exclusive with Variant.
const (
	// PromptGuardProtocolHTTPChat calls an external model through a
	// generative chat-completion prompt (e.g. Qwen3Guard-style).
	PromptGuardProtocolHTTPChat = "http_chat"
	// PromptGuardProtocolHTTPClassify calls an external model through a
	// lightweight sequence-classifier HTTP contract (text in, full
	// label/score distribution out).
	PromptGuardProtocolHTTPClassify = "http_classify"
)

// PromptGuardConfig.OnError values live in classifier_on_error.go as
// OnErrorAllow/OnErrorBlock - shared with every other pluggable classifier
// backend (CategoryModel, PIIModel, ClassifierSignalRule), not just prompt
// guard.

// Signal type constants for rule conditions.
const (
	SignalTypeKeyword      = "keyword"
	SignalTypeEmbedding    = "embedding"
	SignalTypeDomain       = "domain"
	SignalTypeFactCheck    = "fact_check"
	SignalTypeUserFeedback = "user_feedback"
	SignalTypeReask        = "reask"
	SignalTypePreference   = "preference"
	SignalTypeLanguage     = "language"
	SignalTypeContext      = "context"
	SignalTypeStructure    = "structure"
	SignalTypeComplexity   = "complexity"
	SignalTypeModality     = "modality"
	SignalTypeAuthz        = "authz"
	SignalTypeJailbreak    = "jailbreak"
	SignalTypePII          = "pii"
	SignalTypeKB           = "kb"
	SignalTypeConversation = "conversation"
	SignalTypeEvent        = "event"
	SignalTypeProjection   = "projection"
)

// RouterConfig represents the main configuration for the LLM Router.
type RouterConfig struct {
	CanonicalVersion string `yaml:"-" json:"-"`
	BillingCurrency  string `yaml:"-" json:"-"`
	// RoutingSnapshot is the immutable compiled Model/Recipe/Entrypoint value
	// used by BackendInvoker. It is built once at the standalone manifest
	// boundary or supplied by the managed publication replica. Runtime routing
	// views are derived from it and never act as physical-backend authority.
	RoutingSnapshot *routingsnapshot.Snapshot `yaml:"-" json:"-"`
	ControlPlane    ControlPlaneConfig        `yaml:"control_plane,omitempty"`
	MoMRegistry     map[string]string         `yaml:"mom_registry,omitempty"`
	// SkipExternalAssetValidation is set only for untrusted read-only
	// validation requests, which must never trigger filesystem reads.
	SkipExternalAssetValidation bool `yaml:"-" json:"-"`

	// Static global configuration.
	InlineModels     `yaml:",inline"`
	ExternalModels   []ExternalModelConfig `yaml:"external_models,omitempty"`
	SemanticCache    `yaml:"semantic_cache"`
	Memory           MemoryConfig        `yaml:"memory"`
	VectorStore      *VectorStoreConfig  `yaml:"vector_store,omitempty"`
	ResponseAPI      ResponseAPIConfig   `yaml:"response_api"`
	Agent            AgentServiceConfig  `yaml:"agent,omitempty"`
	RouterReplay     RouterReplayConfig  `yaml:"router_replay"`
	StartupStatus    StartupStatusConfig `yaml:"startup_status"`
	Looper           LooperConfig        `yaml:"looper,omitempty"`
	LLMObservability `yaml:",inline"`
	APIServer        `yaml:",inline"`
	RouterOptions    `yaml:",inline"`
	RouterLearning   RouterLearningConfig `yaml:"learning,omitempty"`

	// Dynamic routing configuration. Entrypoints and Recipes are the root
	// runtime authority produced by the canonical loader. The inline
	// IntelligentRouting fields are populated only on an isolated Recipe view;
	// they never define an implicit root-level Recipe.
	IntelligentRouting `yaml:",inline"`
	Entrypoints        []EntrypointMapping `yaml:"-"`
	Recipes            []RoutingRecipe     `yaml:"-"`
	// RoutingScope is populated only on isolated Recipe and Recipe-document
	// views. An empty scope identifies the root runtime config.
	RoutingScope  RecipeName `yaml:"-"`
	BackendModels `yaml:",inline"`
	ToolSelection `yaml:",inline"`

	ManagementAPI      ManagementAPIConfig       `yaml:"management_api,omitempty"`
	Access             AccessServiceConfig       `yaml:"access,omitempty"`
	AccessStore        *AccessStoreConfig        `yaml:"access_store,omitempty"`
	AccessRuntimeStore *AccessRuntimeStoreConfig `yaml:"access_runtime_store,omitempty"`
	BackendCredentials BackendCredentialsConfig  `yaml:"backend_credentials,omitempty"`
	BackendEgress      BackendEgressConfig       `yaml:"backend_egress,omitempty"`
	BackendDispatch    BackendDispatchConfig     `yaml:"backend_dispatch,omitempty"`

	// Runtime-only knowledge bases loaded from global.model_catalog.
	KnowledgeBases []KnowledgeBaseConfig `yaml:"knowledge_bases,omitempty"`
	ConfigBaseDir  string                `yaml:"-"`
	// DocumentHash identifies the exact YAML document from which this immutable
	// runtime snapshot was parsed. Management APIs use it to distinguish a
	// persisted config from the config that has completed hot reload.
	DocumentHash string `yaml:"-"`
}

type ToolSelection struct {
	Tools ToolsConfig `yaml:"tools"`
}

type Listener struct {
	Name    string `yaml:"name"`
	Address string `yaml:"address"`
	Port    int    `yaml:"port"`
	Timeout string `yaml:"timeout,omitempty"`
}

type APIServer struct {
	Listeners []Listener `yaml:"listeners,omitempty"`
	API       APIConfig  `yaml:"api"`
}

type LLMObservability struct {
	Observability ObservabilityConfig `yaml:"observability"`
}

type RouterOptions struct {
	IncludeConfigModelsInList bool  `yaml:"include_config_models_in_list,omitempty"`
	ClearRouteCache           bool  `yaml:"clear_route_cache"`
	StreamedBodyMode          bool  `yaml:"streamed_body_mode,omitempty"`
	MaxStreamedBodyBytes      int64 `yaml:"max_streamed_body_bytes,omitempty"`
	StreamedBodyTimeoutSec    int   `yaml:"streamed_body_timeout_sec,omitempty"`
}

// InlineModels captures built-in model families and prompt-processing settings.
type InlineModels struct {
	EmbeddingModels         `yaml:"embedding_models"`
	Classifier              `yaml:"classifier"`
	ComplexityModel         ComplexityModelConfig         `yaml:"complexity_model,omitempty"`
	PromptCompression       PromptCompressionConfig       `yaml:"prompt_compression"`
	PromptGuard             PromptGuardConfig             `yaml:"prompt_guard"`
	HallucinationMitigation HallucinationMitigationConfig `yaml:"hallucination_mitigation"`
	FeedbackDetector        FeedbackDetectorConfig        `yaml:"feedback_detector"`
	ModalityDetector        ModalityDetectorConfig        `yaml:"modality_detector"`
}

// IntelligentRouting captures user-facing signal and decision configuration.
type IntelligentRouting struct {
	Signals         `yaml:",inline"`
	Projections     Projections          `yaml:"projections,omitempty"`
	Decisions       []Decision           `yaml:"decisions,omitempty"`
	Strategy        RoutingStrategy      `yaml:"strategy,omitempty"`
	ModelSelection  ModelSelectionConfig `yaml:"model_selection,omitempty"`
	ReasoningConfig `yaml:",inline"`
}

// BackendModels captures configured backend endpoints and model metadata.
type BackendModels struct {
	ModelConfig      map[string]ModelParams          `yaml:"model_config"`
	DefaultModel     string                          `yaml:"default_model"`
	VLLMEndpoints    []VLLMEndpoint                  `yaml:"vllm_endpoints"`
	ImageGenBackends map[string]ImageGenBackendEntry `yaml:"image_gen_backends,omitempty"`
}

type ReasoningConfig struct {
	DefaultReasoningEffort string                           `yaml:"default_reasoning_effort,omitempty"`
	ReasoningFamilies      map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`
}
