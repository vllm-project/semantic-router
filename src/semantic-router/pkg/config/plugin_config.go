package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	ResponseCacheModeSemantic          = "semantic"
	ResponseCacheModeExact             = "exact"
	ResponseCacheModeExactThenSemantic = "exact_then_semantic"
)

// DecisionPlugin represents a plugin configuration for a decision.
// Type is the plugin identifier; the authoritative supported set is registered in
// routing_surface_catalog (and DSL/compiler surfaces), not duplicated here.
type DecisionPlugin struct {
	Type string `yaml:"type" json:"type" jsonschema:"required"`

	// Configuration stores the plugin payload as normalized structured bytes.
	Configuration *StructuredPayload `yaml:"configuration,omitempty" json:"configuration,omitempty" jsonschema:"required"`
}

// ResponseCacheSemanticConfig controls the semantic tier: a request reuses
// the answer of a cached request that asks the same thing.
type ResponseCacheSemanticConfig struct {
	// Match finds the cached requests a request may reuse: recall bounds
	// their similarity, and a same_request judge confirms them. Without one
	// the response cache's default runs: the 3 most similar at 0.8 or more,
	// each confirmed by same_request on the decision model.
	Match *MatchConfig `json:"match,omitempty" yaml:"match,omitempty"`
}

// ResponseCacheRequestControlsConfig authorizes bounded request-level cache directives.
type ResponseCacheRequestControlsConfig struct {
	Enabled       bool     `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	Header        string   `json:"header,omitempty" yaml:"header,omitempty"`
	Allowed       []string `json:"allowed,omitempty" yaml:"allowed,omitempty"`
	MaxTTLSeconds *int     `json:"max_ttl_seconds,omitempty" yaml:"max_ttl_seconds,omitempty"`
}

// ResponseCachePersonalizedConfig controls post-enrichment response caching.
type ResponseCachePersonalizedConfig struct {
	Mode string `json:"mode,omitempty" yaml:"mode,omitempty"`
}

type ResponseCacheRevisionConfig struct {
	CacheEpoch     string `json:"cache_epoch,omitempty" yaml:"cache_epoch,omitempty"`
	ModelRevision  string `json:"model_revision,omitempty" yaml:"model_revision,omitempty"`
	PromptRevision string `json:"prompt_revision,omitempty" yaml:"prompt_revision,omitempty"`
	PolicyRevision string `json:"policy_revision,omitempty" yaml:"policy_revision,omitempty"`
}

// ResponseCachePluginConfig represents route-local response reuse policy.
type ResponseCachePluginConfig struct {
	Enabled         bool                                `json:"enabled" yaml:"enabled"`
	Mode            string                              `json:"mode,omitempty" yaml:"mode,omitempty"`
	Scope           string                              `json:"scope,omitempty" yaml:"scope,omitempty"`
	TTLSeconds      *int                                `json:"ttl_seconds,omitempty" yaml:"ttl_seconds,omitempty"`
	Semantic        *ResponseCacheSemanticConfig        `json:"semantic,omitempty" yaml:"semantic,omitempty"`
	RequestControls *ResponseCacheRequestControlsConfig `json:"request_controls,omitempty" yaml:"request_controls,omitempty"`
	Personalized    *ResponseCachePersonalizedConfig    `json:"personalized,omitempty" yaml:"personalized,omitempty"`
	Revision        *ResponseCacheRevisionConfig        `json:"revision,omitempty" yaml:"revision,omitempty"`
}

// SemanticMatch is the match block the semantic tier runs as written; nil
// runs the response cache's default.
func (c *ResponseCachePluginConfig) SemanticMatch() *MatchConfig {
	if c == nil || c.Semantic == nil {
		return nil
	}
	return c.Semantic.Match
}

// EffectiveRequestControls are the request-level cache directives the plugin
// allows; none without request_controls.
func (c *ResponseCachePluginConfig) EffectiveRequestControls() ResponseCacheRequestControlsConfig {
	if c == nil || c.RequestControls == nil {
		return ResponseCacheRequestControlsConfig{}
	}
	return *c.RequestControls
}

// FastResponsePluginConfig represents configuration for fast_response plugin.
type FastResponsePluginConfig struct {
	// Enabled false turns off the fast_response a decision inherits.
	Enabled *bool  `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	Message string `json:"message" yaml:"message"`
}

// IsEnabled reports whether the fast_response runs: unless enabled is false.
func (c *FastResponsePluginConfig) IsEnabled() bool {
	return c != nil && (c.Enabled == nil || *c.Enabled)
}

// RequestParamsPluginConfig represents configuration for request_params plugin.
// This plugin validates and strips request body parameters per decision.
type RequestParamsPluginConfig struct {
	// DefaultMaxTokens supplies an output bound only when the caller omits it.
	DefaultMaxTokens *OutputTokenDefault `json:"default_max_tokens,omitempty" yaml:"default_max_tokens,omitempty"`
	// BlockedParams is a list of parameters that should be blocked/stripped.
	BlockedParams []string `json:"blocked_params,omitempty" yaml:"blocked_params,omitempty"`
	// MaxTokensLimit sets the maximum allowed value for max_tokens.
	MaxTokensLimit *int `json:"max_tokens_limit,omitempty" yaml:"max_tokens_limit,omitempty"`
	// MaxN is the maximum allowed value for n (number of completions).
	MaxN *int `json:"max_n,omitempty" yaml:"max_n,omitempty"`
}

// How system_prompt applies its prompt: insert puts it before the first
// system block's content, replace takes the place of every system and
// developer block.
const (
	SystemPromptModeInsert  = "insert"
	SystemPromptModeReplace = "replace"
)

// SystemPromptPluginConfig represents configuration for system_prompt plugin.
type SystemPromptPluginConfig struct {
	Enabled      *bool  `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	SystemPrompt string `json:"system_prompt,omitempty" yaml:"system_prompt,omitempty"`
	Mode         string `json:"mode,omitempty" yaml:"mode,omitempty" jsonschema:"enum=insert,enum=replace"`
}

// IsEnabled preserves the implicit activation of a nonempty system prompt.
func (c *SystemPromptPluginConfig) IsEnabled() bool {
	if c == nil {
		return false
	}
	if c.Enabled != nil {
		return *c.Enabled
	}
	return c.SystemPrompt != ""
}

func (c *SystemPromptPluginConfig) EffectiveMode() string {
	if c == nil || c.Mode == "" {
		return SystemPromptModeInsert
	}
	return c.Mode
}

func validateSystemPromptPlugin(at PluginAt, p *SystemPromptPluginConfig) error {
	switch p.Mode {
	case "", SystemPromptModeInsert, SystemPromptModeReplace:
		return nil
	}
	return at.Errorf("mode must be insert or replace")
}

// HeaderMutationPluginConfig represents configuration for header_mutation plugin.
type HeaderMutationPluginConfig struct {
	Add    []HeaderPair `json:"add,omitempty" yaml:"add,omitempty"`
	Update []HeaderPair `json:"update,omitempty" yaml:"update,omitempty"`
	Delete []string     `json:"delete,omitempty" yaml:"delete,omitempty"`
}

// HeaderPair represents a header name-value pair.
type HeaderPair struct {
	Name  string `json:"name" yaml:"name"`
	Value string `json:"value" yaml:"value"`
}

// RouterReplayPluginConfig represents configuration for router_replay plugin.
type RouterReplayPluginConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`
	// PromptCapture is what a record keeps of the request and its answer
	// besides routing data: none, raw or verified (PromptCaptureMode). Unset
	// is none.
	PromptCapture       string `json:"prompt_capture,omitempty" yaml:"prompt_capture,omitempty" jsonschema:"enum=none,enum=raw,enum=verified"`
	MaxRecords          int    `json:"max_records,omitempty" yaml:"max_records,omitempty"`
	CaptureRequestBody  bool   `json:"capture_request_body,omitempty" yaml:"capture_request_body,omitempty"`
	CaptureResponseBody bool   `json:"capture_response_body,omitempty" yaml:"capture_response_body,omitempty"`
	// CaptureToolResults keeps client tool-result text in the captured request
	// body and in each client_tool_result step's Output, bounded by
	// MaxToolTraceBytes. It stores nothing unless PromptCapture is raw and
	// CaptureRequestBody is on, and principals without replay.detail still
	// read the results redacted.
	CaptureToolResults bool `json:"capture_tool_results,omitempty" yaml:"capture_tool_results,omitempty"`
	MaxBodyBytes       int  `json:"max_body_bytes,omitempty" yaml:"max_body_bytes,omitempty"`
	// MaxPromptBytes caps the prompt a record keeps under raw and verified,
	// cut on a UTF-8 boundary; a cut record keeps the whole prompt's size and
	// SHA-256. 0 means no limit. Nil where no layer sets it, so that
	// PromptByteLimit can fall back to MaxToolTraceBytes.
	MaxPromptBytes *int `json:"max_prompt_bytes,omitempty" yaml:"max_prompt_bytes,omitempty" jsonschema:"minimum=0"`
	// MaxToolTraceBytes caps each structured tool text field
	// (ToolDefinitions, ToolTraceStep.Text, Arguments and Output).
	// 0 means no limit. Configurable independently of MaxBodyBytes so that
	// structured fields remain complete even when the raw body is truncated.
	MaxToolTraceBytes int `json:"max_tool_trace_bytes,omitempty" yaml:"max_tool_trace_bytes,omitempty"`
	// MaxToolTraceSteps caps the number of ToolTraceStep entries retained per
	// record. Long agent sessions produce one step per tool-call turn, so an
	// unbounded chain (e.g. a runaway hermes-style loop) can otherwise grow
	// without limit and OOM the router process. When the cap is exceeded the
	// oldest steps are dropped and ToolTrace.StepsTruncated is set so callers
	// can see that a cap was applied. 0 means no limit.
	MaxToolTraceSteps int `json:"max_tool_trace_steps,omitempty" yaml:"max_tool_trace_steps,omitempty"`
	// Shadow sends a sampled copy of the decision's requests to a shadow
	// model, whose outcome the replay record keeps. It runs while the
	// decision records replays.
	Shadow *RouterReplayShadowConfig `json:"shadow,omitempty" yaml:"shadow,omitempty"`
	// CapturePersonalData is the deprecated form of PromptCapture: false
	// meant verified and true raw. Loading a configuration rewrites it as
	// prompt_capture in its own layer, with a warning; in a policy assembled
	// otherwise it decides only where PromptCapture is unset.
	CapturePersonalData *bool `json:"capture_personal_data,omitempty" yaml:"capture_personal_data,omitempty"`
}

// RouterReplayShadowConfig is router_replay's shadow: a bounded, sampled copy
// of the approved provider-bound request, sent to a secondary configured
// model after the primary dispatch has been finalized. The primary response
// never waits on, or changes because of, the shadow call.
type RouterReplayShadowConfig struct {
	// Model is the configured logical model that receives the shadow copy.
	Model string `json:"model" yaml:"model"`
	// SampleRate is the fraction of eligible requests that are shadowed, in
	// [0, 1]. nil means every eligible request; 0 disables dispatch while
	// keeping the shadow declared.
	SampleRate *float64 `json:"sample_rate,omitempty" yaml:"sample_rate,omitempty"`
	// MaxConcurrency bounds in-flight shadow calls for this decision.
	MaxConcurrency int `json:"max_concurrency,omitempty" yaml:"max_concurrency,omitempty"`
	// MaxQueueDepth bounds shadow calls waiting for an in-flight slot. Calls
	// beyond the depth are dropped with a queue_full reason.
	MaxQueueDepth int `json:"max_queue_depth,omitempty" yaml:"max_queue_depth,omitempty"`
	// TimeoutSeconds bounds queue wait plus execution for one shadow call.
	TimeoutSeconds int `json:"timeout_seconds,omitempty" yaml:"timeout_seconds,omitempty"`
	// MaxResponseBytes bounds the shadow response body read from the backend.
	MaxResponseBytes int `json:"max_response_bytes,omitempty" yaml:"max_response_bytes,omitempty"`
	// MaxRetries bounds additional attempts on transport errors or retryable
	// upstream statuses. All attempts share the same deadline.
	MaxRetries int `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`
	// CaptureResponseBody stores a bounded excerpt of the shadow output text
	// in the replay outcome when the request's prompt_capture is raw, which
	// alone keeps generated text. Off by default: only hashes and sizes are
	// kept.
	CaptureResponseBody bool `json:"capture_response_body,omitempty" yaml:"capture_response_body,omitempty"`
	// MaxCaptureBytes bounds the stored excerpt when CaptureResponseBody is on.
	MaxCaptureBytes int `json:"max_capture_bytes,omitempty" yaml:"max_capture_bytes,omitempty"`
	// TLSSkipVerify disables certificate verification for an https shadow
	// backend. The primary path reaches backends through Envoy, which does not
	// verify upstream certificates, so this matches that posture for internal
	// CAs. Off by default.
	TLSSkipVerify bool `json:"tls_skip_verify,omitempty" yaml:"tls_skip_verify,omitempty"`
	// ForwardHeaders names the decision header_mutation headers a shadow copy
	// may carry. Nothing a decision sets for the primary backend is forwarded
	// unless listed here, so a custom credential such as X-Internal-Token
	// stays on the primary path. Known credential carriers (Authorization,
	// Proxy-Authorization, Cookie, x-api-key, api-key, x-goog-api-key, the
	// x-user-*-key headers) cannot be listed. Names match case-insensitively.
	// Empty by default.
	ForwardHeaders []string `json:"forward_headers,omitempty" yaml:"forward_headers,omitempty"`
}

// GetPlugin returns the plugin entry for a specific plugin type.
func (d *Decision) GetPlugin(pluginType string) *DecisionPlugin {
	normalizedTarget := NormalizeDecisionPluginType(pluginType)
	for i := range d.Plugins {
		if NormalizeDecisionPluginType(d.Plugins[i].Type) == normalizedTarget {
			return &d.Plugins[i]
		}
	}
	return nil
}

// HasPlugin reports whether the decision includes a plugin of the given type.
func (d *Decision) HasPlugin(pluginType string) bool {
	return d.GetPlugin(pluginType) != nil
}

// UnmarshalPluginConfig converts a plugin payload into the given target struct.
func UnmarshalPluginConfig(config *StructuredPayload, target interface{}) error {
	if config == nil {
		return fmt.Errorf("plugin configuration is nil")
	}
	return config.DecodeInto(target)
}

// GetResponseCacheConfig returns route-local response cache settings.
func (d *Decision) GetResponseCacheConfig() *ResponseCachePluginConfig {
	result := &ResponseCachePluginConfig{}
	return decodeDecisionPlugin(d, DecisionPluginResponseCache, result)
}

// GetSystemPromptConfig returns the system_prompt plugin configuration.
func (d *Decision) GetSystemPromptConfig() *SystemPromptPluginConfig {
	return SystemPromptPluginOf(d.Plugins)
}

// SystemPromptPluginOf is the system_prompt plugin among a request's
// plugins, nil without one.
func SystemPromptPluginOf(plugins []DecisionPlugin) *SystemPromptPluginConfig {
	return decodePlugin(plugins, DecisionPluginSystemPrompt, &SystemPromptPluginConfig{})
}

// GetHeaderMutationConfig returns the header_mutation plugin configuration.
func (d *Decision) GetHeaderMutationConfig() *HeaderMutationPluginConfig {
	return HeaderMutationPluginOf(d.Plugins)
}

// HeaderMutationPluginOf is the header_mutation plugin among a request's
// plugins, nil without one.
func HeaderMutationPluginOf(plugins []DecisionPlugin) *HeaderMutationPluginConfig {
	return decodePlugin(plugins, DecisionPluginHeaderMutation, &HeaderMutationPluginConfig{})
}

// GetRouterReplayConfig returns the router_replay plugin configuration.
func (d *Decision) GetRouterReplayConfig() *RouterReplayPluginConfig {
	result := &RouterReplayPluginConfig{}
	return decodeDecisionPlugin(d, "router_replay", result)
}

// GetFastResponseConfig returns the fast_response plugin configuration.
func (d *Decision) GetFastResponseConfig() *FastResponsePluginConfig {
	return FastResponsePluginOf(d.Plugins)
}

// FastResponsePluginOf is the fast_response plugin among a request's plugins,
// nil without one or when disabled.
func FastResponsePluginOf(plugins []DecisionPlugin) *FastResponsePluginConfig {
	plugin := decodePlugin(plugins, DecisionPluginFastResponse, &FastResponsePluginConfig{})
	if !plugin.IsEnabled() {
		return nil
	}
	return plugin
}

// GetRequestParamsConfig returns the request_params plugin configuration.
func (d *Decision) GetRequestParamsConfig() *RequestParamsPluginConfig {
	return RequestParamsPluginOf(d.Plugins)
}

// RequestParamsPluginOf is the request_params plugin among a request's
// plugins, nil without one.
func RequestParamsPluginOf(plugins []DecisionPlugin) *RequestParamsPluginConfig {
	return decodePlugin(plugins, DecisionPluginRequestParams, &RequestParamsPluginConfig{})
}

func decodeDecisionPlugin[T any](d *Decision, pluginType string, result *T) *T {
	plugin := d.GetPlugin(pluginType)
	if plugin == nil || plugin.Configuration == nil {
		return nil
	}

	if err := UnmarshalPluginConfig(plugin.Configuration, result); err != nil {
		logging.Errorf("Failed to unmarshal %s config: %v", pluginType, err)
		return nil
	}
	return result
}
