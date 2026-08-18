package config

// ConfigSource defines where to load dynamic configuration from.
type ConfigSource string

const (
	// ConfigSourceFile loads configuration from file (default).
	ConfigSourceFile ConfigSource = "file"
	// ConfigSourceKubernetes loads configuration from Kubernetes CRDs.
	ConfigSourceKubernetes ConfigSource = "kubernetes"
)

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

// PromptGuardConfig.OnError values, selecting what a classifier-backend
// failure (e.g. an unreachable http_chat/http_classify endpoint) does to the
// jailbreak rule that failed to evaluate. Deliberately NOT named/valued
// after this package's other on_error: skip|fail fields (fusion/remom/
// workflows/confidence configs) - there, "fail" means "propagate the error
// and halt"; here it means "treat as a positive detection and block", a
// different behavior that the same word would misleadingly imply is the
// same. allow/block names the actual effect on the request instead.
const (
	// PromptGuardOnErrorAllow preserves the historical behavior: a classify
	// error is logged and the affected content is treated as not matching
	// this rule, so other content/rules still evaluate normally. This is the
	// default when OnError is unset.
	PromptGuardOnErrorAllow = "allow"
	// PromptGuardOnErrorBlock treats a classify error as if the rule matched
	// at maximum confidence - fail-closed, because an inference failure
	// means the content could not be verified safe. Without this, an
	// unreachable classifier endpoint looks identical to a genuinely clean
	// request (see @adaamko's review on #2760).
	PromptGuardOnErrorBlock = "block"
)

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

// API format constants for model backends.
const (
	APIFormatOpenAI    = "openai"
	APIFormatAnthropic = "anthropic"
)

// ClientProtocol* identifies the inbound wire format; distinct from APIFormat (upstream backend).
// The zero value (empty string) is treated as OpenAI-compatible; additional constants will be
// introduced as follow-up changes add explicit consumers.
const (
	ClientProtocolAnthropic = "anthropic"
)

// RouterConfig represents the main configuration for the LLM Router.
type RouterConfig struct {
	ConfigSource ConfigSource      `yaml:"config_source,omitempty"`
	MoMRegistry  map[string]string `yaml:"mom_registry,omitempty"`
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
	RouterReplay     RouterReplayConfig  `yaml:"router_replay"`
	StartupStatus    StartupStatusConfig `yaml:"startup_status"`
	Looper           LooperConfig        `yaml:"looper,omitempty"`
	LLMObservability `yaml:",inline"`
	APIServer        `yaml:",inline"`
	RouterOptions    `yaml:",inline"`
	RouterLearning   RouterLearningConfig `yaml:"learning,omitempty"`

	// Dynamic user-facing routing configuration. Entrypoints and Recipes are
	// the normalized multi-recipe state produced by the canonical loader; the
	// inline IntelligentRouting fields always mirror the default recipe.
	IntelligentRouting `yaml:",inline"`
	Entrypoints        []EntrypointMapping `yaml:"-"`
	Recipes            []RoutingRecipe     `yaml:"-"`
	// RoutingScope is populated only on immutable recipe views.
	RoutingScope  RecipeName `yaml:"-"`
	BackendModels `yaml:",inline"`
	ToolSelection `yaml:",inline"`

	Authz         AuthzConfig         `yaml:"authz,omitempty"`
	RateLimit     RateLimitConfig     `yaml:"ratelimit,omitempty"`
	ManagementAPI ManagementAPIConfig `yaml:"management_api,omitempty"`

	// Runtime-only knowledge bases loaded from global.model_catalog.
	KnowledgeBases []KnowledgeBaseConfig `yaml:"knowledge_bases,omitempty"`
	ConfigBaseDir  string                `yaml:"-"`
	// DocumentHash identifies the exact YAML document from which this immutable
	// runtime snapshot was parsed. Management APIs use it to distinguish a
	// persisted config from the config that has completed hot reload.
	DocumentHash string `yaml:"-"`
}

// AuthzConfig configures how the router resolves per-user LLM API keys.
type AuthzConfig struct {
	FailOpen  bool                  `yaml:"fail_open,omitempty"`
	Identity  IdentityConfig        `yaml:"identity,omitempty"`
	Providers []AuthzProviderConfig `yaml:"providers,omitempty"`
}

// IdentityConfig controls how the router reads user identity from request headers.
type IdentityConfig struct {
	UserIDHeader     string `yaml:"user_id_header,omitempty"`
	UserGroupsHeader string `yaml:"user_groups_header,omitempty"`
}

func (ic IdentityConfig) GetUserIDHeader() string {
	if ic.UserIDHeader == "" {
		return "x-authz-user-id"
	}
	return ic.UserIDHeader
}

func (ic IdentityConfig) GetUserGroupsHeader() string {
	if ic.UserGroupsHeader == "" {
		return "x-authz-user-groups"
	}
	return ic.UserGroupsHeader
}

type AuthzProviderConfig struct {
	Type    string            `yaml:"type"`
	Headers map[string]string `yaml:"headers,omitempty"`
}

type RateLimitConfig struct {
	FailOpen  bool                      `yaml:"fail_open,omitempty"`
	Providers []RateLimitProviderConfig `yaml:"providers,omitempty"`
}

type RateLimitProviderConfig struct {
	Type    string          `yaml:"type"`
	Address string          `yaml:"address,omitempty"`
	Domain  string          `yaml:"domain,omitempty"`
	Rules   []RateLimitRule `yaml:"rules,omitempty"`
}

type RateLimitRule struct {
	Name            string         `yaml:"name"`
	Match           RateLimitMatch `yaml:"match"`
	RequestsPerUnit int            `yaml:"requests_per_unit,omitempty"`
	TokensPerUnit   int            `yaml:"tokens_per_unit,omitempty"`
	Unit            string         `yaml:"unit"`
}

type RateLimitMatch struct {
	User  string `yaml:"user,omitempty"`
	Group string `yaml:"group,omitempty"`
	Model string `yaml:"model,omitempty"`
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
	AutoModelName             string               `yaml:"auto_model_name,omitempty"`
	AutoModelNames            []string             `yaml:"auto_model_names,omitempty"`
	IncludeConfigModelsInList bool                 `yaml:"include_config_models_in_list,omitempty"`
	ClearRouteCache           bool                 `yaml:"clear_route_cache"`
	StreamedBodyMode          bool                 `yaml:"streamed_body_mode,omitempty"`
	MaxStreamedBodyBytes      int64                `yaml:"max_streamed_body_bytes,omitempty"`
	StreamedBodyTimeoutSec    int                  `yaml:"streamed_body_timeout_sec,omitempty"`
	SkipProcessing            SkipProcessingConfig `yaml:"skip_processing,omitempty"`
}

// SkipProcessingConfig gates the x-vsr-skip-processing request header.
type SkipProcessingConfig struct {
	Enabled bool `yaml:"enabled"`
}

// IsEnabled reports whether the x-vsr-skip-processing opt-out is honored.
func (s SkipProcessingConfig) IsEnabled() bool {
	return s.Enabled
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
	ProviderProfiles map[string]ProviderProfile      `yaml:"provider_profiles,omitempty"`
}

type ReasoningConfig struct {
	DefaultReasoningEffort string                           `yaml:"default_reasoning_effort,omitempty"`
	ReasoningFamilies      map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`
}
