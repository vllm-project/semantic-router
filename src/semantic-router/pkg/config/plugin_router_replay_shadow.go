package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// shadowCredentialHeaders names the headers that carry a credential on the
// primary path: the auth headers the provider registry knows, the per-user
// key headers the auth backend injects, and the generic HTTP credential
// carriers. A shadow copy never carries them. Names are lower-case.
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
	maxShadowModels                    = 8
	defaultShadowSampleRate            = 1.0
	defaultShadowMaxConcurrency        = 2
	defaultShadowMaxQueueSize          = 8
	defaultShadowTimeout               = "30s"
	defaultShadowResponseBytesPerModel = 1 << 20
	defaultShadowCaptureBytesPerModel  = 4096
	maxShadowTimeout                   = 5 * time.Minute
	maxShadowTimeoutLength             = 128
)

var shadowDurationPattern = regexp.MustCompile(`^\+?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:ms|s|m)(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:ms|s|m))*$`)

// UnmarshalJSON checks presence before WithDefaults fills in runtime defaults.
// YAML and DSL plugin payloads pass through this same strict JSON decoder.
func (c *RouterReplayShadowConfig) UnmarshalJSON(data []byte) error {
	type plain RouterReplayShadowConfig
	var decoded plain
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for _, name := range []string{"execution", "budget", "capture"} {
		if isNullJSON(fields[name]) {
			return fmt.Errorf("%s must be an object when provided", name)
		}
	}
	var execution struct {
		Timeout json.RawMessage `json:"timeout"`
	}
	if raw, ok := fields["execution"]; ok {
		if err := json.Unmarshal(raw, &execution); err != nil {
			return err
		}
	}
	if len(execution.Timeout) > 0 && decoded.Execution.Timeout == "" {
		return fmt.Errorf("execution.timeout must be a non-empty duration when provided")
	}
	var capture struct {
		Enabled json.RawMessage `json:"enabled"`
	}
	if raw, ok := fields["capture"]; ok {
		if err := json.Unmarshal(raw, &capture); err != nil {
			return err
		}
	}
	if isNullJSON(capture.Enabled) {
		return fmt.Errorf("capture.enabled must be a boolean when provided")
	}
	*c = RouterReplayShadowConfig(decoded)
	return nil
}

// DefaultRouterReplayShadowConfig returns bounded defaults. Models must be
// explicitly configured before enabling observations.
func DefaultRouterReplayShadowConfig() RouterReplayShadowConfig {
	return (RouterReplayShadowConfig{}).WithDefaults()
}

func shadowBound(value *int, fallback int) *int {
	if value != nil {
		fallback = *value
	}
	return &fallback
}

// WithDefaults returns an independent configuration snapshot.
func (c RouterReplayShadowConfig) WithDefaults() RouterReplayShadowConfig {
	c.Models = append([]string(nil), c.Models...)
	c.ForwardHeaders = append([]string(nil), c.ForwardHeaders...)
	for i := range c.Models {
		c.Models[i] = strings.TrimSpace(c.Models[i])
	}
	n := max(1, len(c.Models))
	c.Execution.MaxConcurrency = shadowBound(c.Execution.MaxConcurrency, defaultShadowMaxConcurrency)
	c.Execution.MaxQueueSize = shadowBound(c.Execution.MaxQueueSize, defaultShadowMaxQueueSize)
	if c.Execution.Timeout == "" {
		c.Execution.Timeout = defaultShadowTimeout
	}
	c.Budget.MaxResponseBytes = shadowBound(c.Budget.MaxResponseBytes, n*defaultShadowResponseBytesPerModel)
	c.Capture.MaxBytes = shadowBound(c.Capture.MaxBytes, n*defaultShadowCaptureBytesPerModel)
	if c.SampleRate != nil {
		rate := *c.SampleRate
		c.SampleRate = &rate
	}
	return c
}

func (c *RouterReplayShadowConfig) EffectiveSampleRate() float64 {
	if c == nil || c.SampleRate == nil {
		return defaultShadowSampleRate
	}
	return *c.SampleRate
}

func (c *RouterReplayShadowConfig) Validate() error {
	if c == nil {
		return nil
	}
	if len(c.Models) == 0 {
		return fmt.Errorf("models is required")
	}
	if len(c.Models) > maxShadowModels {
		return fmt.Errorf("models cannot exceed %d candidates", maxShadowModels)
	}
	seen := map[string]bool{}
	for _, model := range c.Models {
		name := strings.TrimSpace(model)
		if name == "" {
			return fmt.Errorf("models cannot contain empty names")
		}
		if seen[name] {
			return fmt.Errorf("models contains duplicate %q", name)
		}
		seen[name] = true
	}
	rate := c.EffectiveSampleRate()
	if math.IsNaN(rate) || math.IsInf(rate, 0) || rate < 0 || rate > 1 {
		return fmt.Errorf("sample_rate must be finite and between 0 and 1")
	}
	effective := c.WithDefaults()
	for _, bound := range []struct {
		name  string
		value *int
	}{
		{"execution.max_concurrency", effective.Execution.MaxConcurrency},
		{"execution.max_queue_size", effective.Execution.MaxQueueSize},
		{"budget.max_response_bytes", effective.Budget.MaxResponseBytes},
		{"capture.max_bytes", effective.Capture.MaxBytes},
	} {
		if *bound.value <= 0 {
			return fmt.Errorf("%s must be positive", bound.name)
		}
	}
	duration, err := time.ParseDuration(effective.Execution.Timeout)
	if err != nil || duration <= 0 || duration > maxShadowTimeout || len(effective.Execution.Timeout) > maxShadowTimeoutLength || !shadowDurationPattern.MatchString(effective.Execution.Timeout) {
		return fmt.Errorf("execution.timeout must be a positive duration at most 5m")
	}
	if *effective.Budget.MaxResponseBytes < 2*len(c.Models) {
		return fmt.Errorf("budget.max_response_bytes must allocate at least two bytes to each model")
	}
	for _, name := range c.ForwardHeaders {
		if err := validateForwardedHeaderName("forward_headers", name); err != nil {
			return err
		}
	}
	return nil
}

// validateDecisionReplayShadow limits shadows to single-model primary dispatch.
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
	return nil
}

// validateReplayShadowModels checks all candidates, including default plugins.
func validateReplayShadowModels(cfg *RouterConfig, owner string, shadow *RouterReplayShadowConfig) error {
	if shadow == nil {
		return nil
	}
	for _, model := range shadow.WithDefaults().Models {
		if len(cfg.GetEndpointsForModel(model)) == 0 {
			return fmt.Errorf(
				"decision %q: router_replay shadow model %q has no configured backend",
				owner,
				model,
			)
		}
		if path, mode, ok := cfg.CredentialBackendOf(model); ok {
			return requestCredentialRouterOwnError(path, mode, fmt.Sprintf("router_replay.shadow copies (decisions[%s]) call it as the Router", owner))
		}
	}
	return nil
}
