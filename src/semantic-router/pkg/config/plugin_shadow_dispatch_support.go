package config

import (
	"fmt"
	"strings"
)

const (
	defaultShadowDispatchSampleRate       = 1.0
	defaultShadowDispatchMaxConcurrency   = 2
	defaultShadowDispatchMaxQueueDepth    = 8
	defaultShadowDispatchTimeoutSeconds   = 30
	defaultShadowDispatchMaxResponseBytes = 1 << 20
	defaultShadowDispatchMaxCaptureBytes  = 4096
	// maxShadowDispatchRetries keeps the total resource use of one shadow
	// observation bounded even when an operator asks for retries.
	maxShadowDispatchRetries = 3
)

// DefaultShadowDispatchPluginConfig returns the bounds applied when a
// shadow_dispatch plugin omits a field. Enabled and Model have no default:
// an operator must opt in and name the shadow model explicitly.
func DefaultShadowDispatchPluginConfig() ShadowDispatchPluginConfig {
	return ShadowDispatchPluginConfig{
		MaxConcurrency:   defaultShadowDispatchMaxConcurrency,
		MaxQueueDepth:    defaultShadowDispatchMaxQueueDepth,
		TimeoutSeconds:   defaultShadowDispatchTimeoutSeconds,
		MaxResponseBytes: defaultShadowDispatchMaxResponseBytes,
		MaxCaptureBytes:  defaultShadowDispatchMaxCaptureBytes,
	}
}

// WithDefaults returns a copy with every zero bound replaced by its default.
// SampleRate is left as declared; use EffectiveSampleRate to read it.
func (c ShadowDispatchPluginConfig) WithDefaults() ShadowDispatchPluginConfig {
	defaults := DefaultShadowDispatchPluginConfig()
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

// EffectiveSampleRate returns the declared sample rate, or 1.0 when omitted.
func (c *ShadowDispatchPluginConfig) EffectiveSampleRate() float64 {
	if c == nil || c.SampleRate == nil {
		return defaultShadowDispatchSampleRate
	}
	return *c.SampleRate
}

// Validate checks the payload in isolation. Both the config loader and the
// Kubernetes converter use it so the two admission paths agree. The
// referenced model is checked against the model catalog separately because
// the payload has no access to the router configuration.
func (c *ShadowDispatchPluginConfig) Validate() error {
	if c == nil {
		return nil
	}
	if c.Enabled && strings.TrimSpace(c.Model) == "" {
		return fmt.Errorf("model is required when enabled")
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
	if c.MaxRetries > maxShadowDispatchRetries {
		return fmt.Errorf("max_retries cannot exceed %d", maxShadowDispatchRetries)
	}
	return nil
}

func validateShadowDispatchPlugin(
	decisionName string,
	index int,
	pluginType string,
	typed *ShadowDispatchPluginConfig,
) error {
	if err := typed.Validate(); err != nil {
		return fmt.Errorf("decision %q plugins[%d] (%s): %w", decisionName, index, pluginType, err)
	}
	return nil
}

// validateDecisionShadowDispatchPlugin checks that an enabled shadow model
// resolves to a configured backend, so a shadow copy can never leave the
// router's trust boundary toward an unknown endpoint. The shadow model may
// also appear in modelRefs: the runtime skips a shadow whenever the primary
// dispatch already selected that same model.
func validateDecisionShadowDispatchPlugin(cfg *RouterConfig, decision *Decision) error {
	if cfg == nil || decision == nil {
		return nil
	}
	shadow := decision.GetShadowDispatchConfig()
	if shadow == nil || !shadow.Enabled {
		return nil
	}
	if decision.Algorithm != nil && IsLooperAlgorithmType(decision.Algorithm.Type) {
		return fmt.Errorf(
			"decision %q: shadow_dispatch is not supported on looper-executed algorithm %q; the shadow hook runs only on single-model provider dispatch",
			decision.Name,
			decision.Algorithm.Type,
		)
	}
	model := strings.TrimSpace(shadow.Model)
	if len(cfg.GetEndpointsForModel(model)) == 0 {
		return fmt.Errorf(
			"decision %q: shadow_dispatch model %q has no configured backend",
			decision.Name,
			model,
		)
	}
	return nil
}
