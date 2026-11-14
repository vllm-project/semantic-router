package config

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiregistry"
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// ConfigSource defines where to load dynamic configuration from.
type ConfigSource string

const (
	// ConfigSourceFile loads configuration from file (default).
	ConfigSourceFile ConfigSource = "file"
	// ConfigSourceKubernetes loads configuration from Kubernetes CRDs.
	ConfigSourceKubernetes ConfigSource = "kubernetes"
)

// Signal type constants for rule conditions.
const (
	SignalTypeKeyword       = "keyword"
	SignalTypeDomain        = "domain"
	SignalTypeFactCheck     = "fact_check"
	SignalTypeUserFeedback  = "user_feedback"
	SignalTypeLanguage      = "language"
	SignalTypeContext       = "context"
	SignalTypeComplexity    = "complexity"
	SignalTypeModality      = "modality"
	SignalTypeAuthz         = "authz"
	SignalTypeJailbreak     = "jailbreak"
	SignalTypeHallucination = "hallucination"
	SignalTypePII           = "pii"
	SignalTypeConversation  = "conversation"
	SignalTypeProjection    = "projection"
)

// API names of model backends (providers.models[].api_format), which are the
// API registry's names. A configuration may still write the aliases openai
// and anthropic; the Router reads them as chat_completions and messages.
const (
	APIFormatSystemOne       = string(apiregistry.SystemOne)
	APIFormatChatCompletions = string(apiregistry.ChatCompletions)
	APIFormatResponses       = string(apiregistry.Responses)
	APIFormatMessages        = string(apiregistry.Messages)
	// APIFormatImages selects the DALL-E-compatible image-generation dialect
	// (/v1/images/generations), used to sink responses hosted image_generation
	// requests to diffusion backends.
	APIFormatImages = string(apiregistry.Images)
	// APIFormatVideos selects the OpenAI Videos job dialect (/v1/videos).
	APIFormatVideos = string(apiregistry.Videos)
	// APIFormatOpenAI and APIFormatAnthropic are aliases of
	// APIFormatChatCompletions and APIFormatMessages.
	APIFormatOpenAI    = "openai"
	APIFormatAnthropic = "anthropic"
)

// CanonicalAPIFormat is the API name an api_format value names: an alias
// reads as its API, and empty or unknown values stay as they are.
func CanonicalAPIFormat(apiFormat string) string {
	if apiFormat == "" {
		return ""
	}
	return apiregistry.Canonical(apiFormat)
}

// ClientProtocol* identifies the inbound wire format; distinct from APIFormat (upstream backend).
// The zero value (empty string) is treated as OpenAI-compatible; additional constants will be
// introduced as follow-up changes add explicit consumers.
const (
	ClientProtocolAnthropic = "anthropic"
)

// RouterConfig represents the main configuration for the LLM Router.
type RouterConfig struct {
	KVTransfer   *KVTransferConfig `yaml:"-"`
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
	RoutingDefaults  RoutingDefaults `yaml:"-" json:"-"`
	InlineModels     `yaml:",inline"`
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
	// GlobalPlugins are global.plugins, the default plugins of every
	// decision and of a request that matches none.
	GlobalPlugins []DecisionPlugin `yaml:"-" json:"-"`
	// RoutingScope is populated only on immutable recipe views.
	RoutingScope RecipeName `yaml:"-"`
	// defaultRecipe names the default recipe; empty is DefaultRecipeName.
	defaultRecipe RecipeName
	BackendModels `yaml:",inline"`

	Authz         AuthzConfig         `yaml:"authz,omitempty"`
	RateLimit     RateLimitConfig     `yaml:"ratelimit,omitempty"`
	ManagementAPI ManagementAPIConfig `yaml:"management_api,omitempty"`

	ConfigBaseDir string `yaml:"-"`
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
	// DelegatedCredential is the source of delegated credentials: the
	// trusted authentication result header that carries the requesting
	// user's credential for upstreams whose auth is credential: delegated.
	DelegatedCredential DelegatedCredentialConfig `yaml:"delegated_credential,omitempty"`
}

// DelegatedCredentialConfig declares the credential a deployment
// authenticator issues for the user of an authenticated request. The Router
// trusts it exactly as it trusts the identity headers: the gateway in front
// removes client-supplied copies, and a standalone listener drops them unless
// it trusts identity headers. The value is presented only to an upstream
// whose auth is credential: delegated, under that upstream's auth header, by
// a call made for the request, and never to any other upstream. Disabled by
// default.
type DelegatedCredentialConfig struct {
	Enabled bool `yaml:"enabled,omitempty"`
	// Header is the request header the authenticator sets; default
	// x-vsr-delegated-credential.
	Header string `yaml:"header,omitempty"`
}

// GetHeader returns the configured header name, lowercased, or the default.
func (c DelegatedCredentialConfig) GetHeader() string {
	if header := strings.ToLower(strings.TrimSpace(c.Header)); header != "" {
		return header
	}
	return headers.VSRDelegatedCredential
}

// IdentityConfig controls how the router reads user identity from request headers.
type IdentityConfig struct {
	UserIDHeader     string `yaml:"user_id_header,omitempty"`
	UserGroupsHeader string `yaml:"user_groups_header,omitempty"`
}

// GetUserIDHeader returns the configured user id header, lowercased as the
// gateway hands request headers to the Router, or the default.
func (ic IdentityConfig) GetUserIDHeader() string {
	if header := strings.ToLower(strings.TrimSpace(ic.UserIDHeader)); header != "" {
		return header
	}
	return headers.AuthzUserID
}

// GetUserGroupsHeader returns the configured user groups header, lowercased,
// or the default.
func (ic IdentityConfig) GetUserGroupsHeader() string {
	if header := strings.ToLower(strings.TrimSpace(ic.UserGroupsHeader)); header != "" {
		return header
	}
	return headers.AuthzUserGroups
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

type Listener struct {
	Name    string `yaml:"name"`
	Address string `yaml:"address"`
	Port    int    `yaml:"port"`
	Timeout string `yaml:"timeout,omitempty"`
	// APIKeys are client bearer credentials the listener enforces, in
	// standalone mode and in the CLI-managed Envoy listener.
	APIKeys []string `yaml:"api_keys,omitempty"`
	// APIs, when set, are the only APIs the listener serves, by API name (an
	// alias reads as its API), each with its schema, models and settings;
	// other paths answer 404. Omitted, the listener serves chat_completions,
	// responses, messages and images with their default schemas, and not
	// systemone.
	APIs ListenerAPIs `yaml:"apis,omitempty"`
	// TLS, when set, makes a standalone Router serve this listener over TLS.
	TLS *ListenerTLS `yaml:"tls,omitempty"`
	// Identity, when set, decides whether the listener keeps the client
	// identity headers its requests carry: a standalone Router does, and so
	// does the Envoy listener the CLI generates for --gateway extproc.
	Identity *ListenerIdentity `yaml:"identity,omitempty"`
}

// ListenerIdentity names the identity sources a listener trusts. It trusts
// none by default: with no authenticator in front of the Router, a client
// could claim any identity, so its identity headers are dropped.
type ListenerIdentity struct {
	// TrustHeaders keeps the identity headers (the x-authz-* set and the
	// names global.services.authz.identity sets) that an authenticating proxy
	// or a trusted application in front of the listener asserts.
	TrustHeaders bool `yaml:"trust_headers,omitempty"`
	// TrustedPeers, when set, keeps those headers only on connections whose
	// peer address is in one of these CIDRs; X-Forwarded-For is never read.
	// Empty, every peer of a listener that trusts headers is trusted. Only a
	// standalone Router checks it; the CLI refuses it for --gateway extproc.
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
	RouterEnabled          *bool                `yaml:"router_enabled,omitempty"`
	ListBackendModels      bool                 `yaml:"list_backend_models,omitempty"`
	ClearRouteCache        bool                 `yaml:"clear_route_cache"`
	StreamedBodyMode       bool                 `yaml:"streamed_body_mode,omitempty"`
	MaxStreamedBodyBytes   int64                `yaml:"max_streamed_body_bytes,omitempty"`
	StreamedBodyTimeoutSec int                  `yaml:"streamed_body_timeout_sec,omitempty"`
	SkipProcessing         SkipProcessingConfig `yaml:"skip_processing,omitempty"`
	AllowLossyTranslation  []string             `yaml:"allow_lossy_translation,omitempty"`
	RoutingInput           RoutingInputConfig   `yaml:"routing_input,omitempty"`
	Headers                RouterHeadersConfig  `yaml:"headers,omitempty"`
	ContextWindow          ContextWindowConfig  `yaml:"context_window,omitempty"`
	DrainTimeoutSeconds    *int                 `yaml:"drain_timeout_seconds,omitempty"`
}

// RoutingEnabled controls the recipe pipeline independently of the frontend
// and its model deployments. Omission preserves the default Router mode.
func (c *RouterConfig) RoutingEnabled() bool {
	return c != nil && (c.RouterEnabled == nil || *c.RouterEnabled)
}

// SkipProcessingConfig gates the x-vsr-skip-processing request header.
type SkipProcessingConfig struct {
	Enabled bool `yaml:"enabled"`
}

// IsEnabled reports whether the x-vsr-skip-processing opt-out is honored.
func (s SkipProcessingConfig) IsEnabled() bool {
	return s.Enabled
}

// InlineModels captures the model catalog: deployments, bindings, admission
// and the decision and embedding models.
type InlineModels struct {
	ModelAdmission       map[string]AdmissionConfig `yaml:"model_admission,omitempty"`
	ModelSignalTimeoutMs int                        `yaml:"model_signal_timeout_ms,omitempty"`
	ModelSignalOverload  *SignalOverloadPolicy      `yaml:"model_signal_overload,omitempty"`
	GlobalModelBindings  map[string]ModelBinding    `yaml:"global_model_bindings,omitempty"`
	ModelDeployments     map[string]ModelDeployment `yaml:"model_deployments,omitempty"`
	// DecisionModel is global.model_catalog.system.decision_model, resolved
	// to its canonical name.
	DecisionModel string `yaml:"decision_model,omitempty"`
	// EmbeddingDeployment is global.model_catalog.embeddings.default's
	// deployment.
	EmbeddingDeployment string `yaml:"embedding_deployment,omitempty"`
	// FormWarnings are the renamed forms the document was written in, which
	// it was read as their canonical forms (config_forms.go).
	FormWarnings []ConfigWarning `yaml:"-" json:"-"`

	// scopedBindings are a recipe scope's own bindings, before the global
	// ones (ConfigForRecipe); nil outside a recipe scope.
	scopedBindings map[string]ModelBinding
}

// IntelligentRouting captures user-facing signal and decision configuration.
type IntelligentRouting struct {
	CandidateRequirements *CandidateRequirements  `yaml:"candidate_requirements,omitempty"`
	ModelBindings         map[string]ModelBinding `yaml:"model_bindings,omitempty"`
	// Plugins are the recipe's declared plugins, the layer between
	// GlobalPlugins and a decision's own; DefaultPlugins are the two layered,
	// what a request that matches no decision runs.
	Plugins         []DecisionPlugin `yaml:"-" json:"-"`
	DefaultPlugins  []DecisionPlugin `yaml:"-" json:"-"`
	Signals         `yaml:",inline"`
	Projections     Projections              `yaml:"projections,omitempty"`
	Decisions       []Decision               `yaml:"decisions,omitempty"`
	Strategy        RoutingStrategy          `yaml:"strategy,omitempty"`
	Fallback        *fallback.FallbackPolicy `yaml:"fallback,omitempty" json:"fallback,omitempty"`
	ModelSelection  ModelSelectionConfig     `yaml:"model_selection,omitempty"`
	ReasoningConfig `yaml:",inline"`
}

// BackendModels captures configured backend endpoints and model metadata.
type BackendModels struct {
	ModelConfig         map[string]ModelParams     `yaml:"model_config"`
	DefaultModel        string                     `yaml:"default_model"`
	DefaultQualityIndex string                     `yaml:"-"`
	VLLMEndpoints       []VLLMEndpoint             `yaml:"vllm_endpoints"`
	ProviderProfiles    map[string]ProviderProfile `yaml:"provider_profiles,omitempty"`
	// DefaultBackendTLS is providers.defaults.tls, already applied to each
	// https backend's provider profile.
	DefaultBackendTLS *ProviderDefaultsTLS `yaml:"-" json:"-"`
	// ProviderModelOrder lists providers.models aliases in authored order.
	// VLLMEndpoints are sorted by alias, but the data plane's default route
	// serves the first authored model that has a backend.
	ProviderModelOrder []string `yaml:"-" json:"-"`
}

type ReasoningConfig struct {
	DefaultReasoningEffort string                           `yaml:"default_reasoning_effort,omitempty"`
	ReasoningFamilies      map[string]ReasoningFamilyConfig `yaml:"reasoning_families,omitempty"`
}
