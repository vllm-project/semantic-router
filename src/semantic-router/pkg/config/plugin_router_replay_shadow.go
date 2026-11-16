package config

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// shadowCredentialHeaders names the headers that carry a credential on the
// primary path: the auth headers the provider registry knows, the per-user
// key headers the auth backend injects, and the generic HTTP credential
// carriers. A shadow copy never carries them, and forward_headers may not
// list them. Names are lower-case.
var shadowCredentialHeaders = map[string]struct{}{
	headers.Authorization:              {},
	"proxy-authorization":              {},
	"cookie":                           {},
	headers.XAPIKey:                    {},
	headers.APIKey:                     {},
	headers.XGoogAPIKey:                {},
	headers.UserOpenAIKey:              {},
	headers.UserAnthropicKey:           {},
	headers.UserAzureOpenAIKey:         {},
	headers.UserBedrockKey:             {},
	headers.UserGeminiKey:              {},
	headers.UserVertexAIKey:            {},
	headers.UserMiniMaxKey:             {},
	headers.UserCloudflareWorkersAIKey: {},
}

// IsShadowCredentialHeader reports whether a header name is a known
// credential carrier that must never travel with a shadow copy.
func IsShadowCredentialHeader(name string) bool {
	_, ok := shadowCredentialHeaders[strings.ToLower(strings.TrimSpace(name))]
	return ok
}

// CredentialCarrierHeaders returns the known credential carriers, sorted, for
// a request that must reach its upstream with no credential but its own.
func CredentialCarrierHeaders() []string {
	names := make([]string, 0, len(shadowCredentialHeaders))
	for name := range shadowCredentialHeaders {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

const (
	defaultShadowSampleRate       = 1.0
	defaultShadowMaxConcurrency   = 2
	defaultShadowMaxQueueDepth    = 8
	defaultShadowTimeoutSeconds   = 30
	defaultShadowMaxResponseBytes = 1 << 20
	defaultShadowMaxCaptureBytes  = 4096
	// maxShadowRetries keeps the total resource use of one shadow
	// observation bounded even when an operator asks for retries.
	maxShadowRetries = 3
)

// DefaultRouterReplayShadowConfig returns the bounds applied when a shadow
// omits a field. Model has no default: an operator names the shadow model
// explicitly.
func DefaultRouterReplayShadowConfig() RouterReplayShadowConfig {
	return RouterReplayShadowConfig{
		MaxConcurrency:   defaultShadowMaxConcurrency,
		MaxQueueDepth:    defaultShadowMaxQueueDepth,
		TimeoutSeconds:   defaultShadowTimeoutSeconds,
		MaxResponseBytes: defaultShadowMaxResponseBytes,
		MaxCaptureBytes:  defaultShadowMaxCaptureBytes,
	}
}

// WithDefaults returns a copy with every zero bound replaced by its default.
// SampleRate is left as declared; use EffectiveSampleRate to read it.
func (c RouterReplayShadowConfig) WithDefaults() RouterReplayShadowConfig {
	defaults := DefaultRouterReplayShadowConfig()
	if c.MaxConcurrency <= 0 {
		c.MaxConcurrency = defaults.MaxConcurrency
	}
	if c.MaxQueueDepth <= 0 {
		c.MaxQueueDepth = defaults.MaxQueueDepth
	}
	if c.TimeoutSeconds <= 0 {
		c.TimeoutSeconds = defaults.TimeoutSeconds
	}
	if c.MaxResponseBytes <= 0 {
		c.MaxResponseBytes = defaults.MaxResponseBytes
	}
	if c.MaxCaptureBytes <= 0 {
		c.MaxCaptureBytes = defaults.MaxCaptureBytes
	}
	c.Model = strings.TrimSpace(c.Model)
	return c
}

// ForwardsHeader reports whether a decision header mutation with this name is
// allowlisted for the shadow copy. Matching is case-insensitive. An empty
// allowlist forwards nothing.
func (c *RouterReplayShadowConfig) ForwardsHeader(name string) bool {
	if c == nil {
		return false
	}
	trimmed := strings.TrimSpace(name)
	for _, allowed := range c.ForwardHeaders {
		if strings.EqualFold(strings.TrimSpace(allowed), trimmed) {
			return true
		}
	}
	return false
}

// EffectiveSampleRate returns the declared sample rate, or 1.0 when omitted.
func (c *RouterReplayShadowConfig) EffectiveSampleRate() float64 {
	if c == nil || c.SampleRate == nil {
		return defaultShadowSampleRate
	}
	return *c.SampleRate
}

// Validate checks the shadow in isolation. Both the config loader and the
// Kubernetes converter use it so the two admission paths agree. The
// referenced model is checked against the model catalog separately because
// the payload has no access to the router configuration.
func (c *RouterReplayShadowConfig) Validate() error {
	if c == nil {
		return nil
	}
	if strings.TrimSpace(c.Model) == "" {
		return fmt.Errorf("model is required")
	}
	if c.SampleRate != nil && (*c.SampleRate < 0 || *c.SampleRate > 1) {
		return fmt.Errorf("sample_rate must be between 0 and 1")
	}
	for _, bound := range []struct {
		name  string
		value int
	}{
		{"max_concurrency", c.MaxConcurrency},
		{"max_queue_depth", c.MaxQueueDepth},
		{"timeout_seconds", c.TimeoutSeconds},
		{"max_response_bytes", c.MaxResponseBytes},
		{"max_retries", c.MaxRetries},
		{"max_capture_bytes", c.MaxCaptureBytes},
	} {
		if bound.value < 0 {
			return fmt.Errorf("%s cannot be negative", bound.name)
		}
	}
	if c.MaxRetries > maxShadowRetries {
		return fmt.Errorf("max_retries cannot exceed %d", maxShadowRetries)
	}
	return validateShadowForwardHeaders(c.ForwardHeaders)
}

// validateShadowForwardHeaders rejects allowlist entries that could never be
// forwarded safely, so the mistake is visible at load time instead of being
// silently dropped on every shadow call.
func validateShadowForwardHeaders(names []string) error {
	for _, name := range names {
		if err := validateForwardedHeaderName("forward_headers", name); err != nil {
			return err
		}
	}
	return nil
}

// validateDecisionReplayShadow checks that a decision's shadow model resolves
// to a configured backend, so a shadow copy can never leave the router's
// trust boundary toward an unknown endpoint, and that the decision dispatches
// to one model. The shadow model may also appear in modelRefs: the runtime
// skips a shadow whenever the primary dispatch already selected that model.
func validateDecisionReplayShadow(cfg *RouterConfig, decision *Decision) error {
	if cfg == nil || decision == nil {
		return nil
	}
	replay := decision.GetRouterReplayConfig()
	if replay == nil || replay.Shadow == nil {
		return nil
	}
	if decision.Algorithm != nil && IsExecutorAlgorithmType(decision.Algorithm.Type) {
		return fmt.Errorf(
			"decision %q: router_replay shadow is not supported on executor algorithm %q; the shadow runs only on single-model provider dispatch",
			decision.Name,
			decision.Algorithm.Type,
		)
	}
	model := strings.TrimSpace(replay.Shadow.Model)
	if len(cfg.GetEndpointsForModel(model)) == 0 {
		return fmt.Errorf(
			"decision %q: router_replay shadow model %q has no configured backend",
			decision.Name,
			model,
		)
	}
	if path, mode, ok := cfg.CredentialBackendOf(model); ok {
		return requestCredentialRouterOwnError(path, mode, fmt.Sprintf("router_replay.shadow copies (decisions[%s]) call it as the Router", decision.Name))
	}
	return nil
}
