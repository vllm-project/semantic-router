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

// Signal type constants for rule conditions.
const (
	SignalTypeKeyword       = "keyword"
	SignalTypeEmbedding     = "embedding"
	SignalTypeDomain        = "domain"
	SignalTypeFactCheck     = "fact_check"
	SignalTypeUserFeedback  = "user_feedback"
	SignalTypeReask         = "reask"
	SignalTypePreference    = "preference"
	SignalTypeLanguage      = "language"
	SignalTypeContext       = "context"
	SignalTypeStructure     = "structure"
	SignalTypeComplexity    = "complexity"
	SignalTypeModality      = "modality"
	SignalTypeAuthz         = "authz"
	SignalTypeJailbreak     = "jailbreak"
	SignalTypePII           = "pii"
	SignalTypeKB            = "kb"
	SignalTypeConversation  = "conversation"
	SignalTypeSessionMetric = "session_metric"
	SignalTypeProjection    = "projection"
)

// API format constants for model backends.
const (
	APIFormatOpenAI    = "openai"
	APIFormatAnthropic = "anthropic"
)

// RouterConfig represents the main configuration for the LLM Router.
type RouterConfig struct {
	ConfigSource ConfigSource      `yaml:"config_source,omitempty"`
	MoMRegistry  map[string]string `yaml:"mom_registry,omitempty"`

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

	// Dynamic user-facing routing configuration.
	IntelligentRouting `yaml:",inline"`
	BackendModels      `yaml:",inline"`
	ToolSelection      `yaml:",inline"`

	Authz     AuthzConfig     `yaml:"authz,omitempty"`
	RateLimit RateLimitConfig `yaml:"ratelimit,omitempty"`

	// Runtime-only knowledge bases loaded from global.model_catalog.
	KnowledgeBases []KnowledgeBaseConfig `yaml:"knowledge_bases,omitempty"`
	ConfigBaseDir  string                `yaml:"-"`
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
	AutoModelName             string `yaml:"auto_model_name,omitempty"`
	IncludeConfigModelsInList bool   `yaml:"include_config_models_in_list,omitempty"`
	ClearRouteCache           bool   `yaml:"clear_route_cache"`
	StreamedBodyMode          bool   `yaml:"streamed_body_mode,omitempty"`
	MaxStreamedBodyBytes      int64  `yaml:"max_streamed_body_bytes,omitempty"`
	StreamedBodyTimeoutSec    int    `yaml:"streamed_body_timeout_sec,omitempty"`
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
	Strategy        string               `yaml:"strategy,omitempty"`
	ModelSelection  ModelSelectionConfig `yaml:"model_selection,omitempty"`
	ReasoningConfig `yaml:",inline"`
	SessionStates   []SessionStateConfig `yaml:"session_states,omitempty"`
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
