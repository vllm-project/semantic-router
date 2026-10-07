package config

import (
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

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

// PromptGuardConfig.OnError values live in classifier_on_error.go as
// OnErrorAllow/OnErrorBlock - shared with every other pluggable classifier
// backend (CategoryModel, PIIModel, ClassifierSignalRule), not just prompt
// guard.

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
	SignalTypeHallucination = "hallucination"
	SignalTypePII           = "pii"
	SignalTypeKB            = "kb"
	SignalTypeConversation  = "conversation"
	SignalTypeEvent         = "event"
	SignalTypeProjection    = "projection"
)

// API format constants for model backends.
const (
	APIFormatOpenAI    = "openai"
	APIFormatResponses = "responses"
	APIFormatAnthropic = "anthropic"
	// APIFormatImages selects the DALL-E-compatible image-generation dialect
	// (/v1/images/generations), used to sink responses hosted image_generation
	// requests to diffusion backends.
	APIFormatImages = "images"
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

	// RoutingFragmentOnly marks a config parsed from a routing-only document,
	// which carries no provider or global state. Validation that depends on
	// providers is skipped for these so DSL fragments stay decompilable, while
	// complete configs still get the full contract.
	RoutingFragmentOnly bool `yaml:"-" json:"-"`

	// Static global configuration.
	InlineModels     `yaml:",inline"`
	ExternalModels   []ExternalModelConfig `yaml:"external_models,omitempty"`
	SemanticCache    `yaml:"semantic_cache"`
	Memory           MemoryConfig            `yaml:"memory"`
	VectorStore      *VectorStoreConfig      `yaml:"vector_store,omitempty"`
	ToolSessions     *ToolSessionStoreConfig `yaml:"tool_sessions,omitempty"`
	ResponseAPI      ResponseAPIConfig       `yaml:"response_api"`
	RouterReplay     RouterReplayConfig      `yaml:"router_replay"`
	StartupStatus    StartupStatusConfig     `yaml:"startup_status"`
	Looper           LooperConfig            `yaml:"looper,omitempty"`
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
	// EffectiveModelRegistry is the immutable catalog/config join used to
	// materialize this runtime snapshot.
	EffectiveModelRegistry *modelcatalog.EffectiveRegistry `yaml:"-" json:"-"`
	// Evaluation preserves operator-authored definitions and measurement
	// records for canonical export. Compiled results live in
	// EffectiveModelRegistry.
	Evaluation *CanonicalEvaluation `yaml:"-" json:"-"`
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
	// APIKeys are client bearer credentials the listener enforces, in
	// standalone mode and in the CLI-managed Envoy listener.
	APIKeys []string `yaml:"api_keys,omitempty"`
	// TLS, when set, makes a standalone Router serve this listener over TLS.
	TLS *ListenerTLS `yaml:"tls,omitempty"`
	// Identity, when set, decides whether a standalone Router keeps the client
	// identity headers that requests on this listener carry.
	Identity *ListenerIdentity `yaml:"identity,omitempty"`
}

// ListenerIdentity names the identity sources a standalone listener trusts.
// It trusts none by default: with no authenticator in front of the Router,
// a client could claim any identity, so its identity headers are dropped.
type ListenerIdentity struct {
	// TrustHeaders keeps the identity headers (the x-authz-* set and the
	// names global.services.authz.identity sets) that an authenticating proxy
	// or a trusted application in front of the listener asserts.
	TrustHeaders bool `yaml:"trust_headers,omitempty"`
	// TrustedPeers, when set, keeps those headers only on connections whose
	// peer address is in one of these CIDRs; X-Forwarded-For is never read.
	// Empty, every peer of a listener that trusts headers is trusted.
	TrustedPeers []string `yaml:"trusted_peers,omitempty"`
}

// ListenerTLS is a listener's server certificate for one-way TLS. Relative
// paths are relative to the configuration's directory.
type ListenerTLS struct {
	CertFile string `yaml:"cert_file"`
	KeyFile  string `yaml:"key_file"`
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
	SafetyModels            SafetyModelsConfig            `yaml:"safety_models"`
	FeedbackDetector        FeedbackDetectorConfig        `yaml:"feedback_detector"`
	ModalityDetector        ModalityDetectorConfig        `yaml:"modality_detector"`
	ModelAdmission          map[string]AdmissionConfig    `yaml:"model_admission,omitempty"`
	GlobalModelBindings     map[string]ModelBinding       `yaml:"global_model_bindings,omitempty"`
	ModelDeployments        map[string]ModelDeployment    `yaml:"model_deployments,omitempty"`
	// DecisionModel is global.model_catalog.system.decision_model, resolved
	// to its canonical name.
	DecisionModel string `yaml:"decision_model,omitempty"`
}

// IntelligentRouting captures user-facing signal and decision configuration.
type IntelligentRouting struct {
	CandidateRequirements *CandidateRequirements  `yaml:"candidate_requirements,omitempty"`
	DataPolicy            *RoutingDataPolicy      `yaml:"data_policy,omitempty"`
	ModelBindings         map[string]ModelBinding `yaml:"model_bindings,omitempty"`
	Signals               `yaml:",inline"`
	Projections           Projections              `yaml:"projections,omitempty"`
	Decisions             []Decision               `yaml:"decisions,omitempty"`
	Strategy              RoutingStrategy          `yaml:"strategy,omitempty"`
	Fallback              *fallback.FallbackPolicy `yaml:"fallback,omitempty" json:"fallback,omitempty"`
	ModelSelection        ModelSelectionConfig     `yaml:"model_selection,omitempty"`
	ReasoningConfig       `yaml:",inline"`
}

// BackendModels captures configured backend endpoints and model metadata.
type BackendModels struct {
	ModelConfig         map[string]ModelParams     `yaml:"model_config"`
	DefaultModel        string                     `yaml:"default_model"`
	DefaultQualityIndex string                     `yaml:"-"`
	VLLMEndpoints       []VLLMEndpoint             `yaml:"vllm_endpoints"`
	ProviderProfiles    map[string]ProviderProfile `yaml:"provider_profiles,omitempty"`
	// ProviderModelOrder lists providers.models aliases in authored order.
	// VLLMEndpoints are sorted by alias, but the data plane's default route
	// serves the first authored model that has a backend.
	ProviderModelOrder []string `yaml:"-" json:"-"`
}

type ReasoningConfig struct {
	DefaultReasoningEffort string                           `yaml:"default_reasoning_effort,omitempty"`
	ReasoningFamilies      map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`
}
