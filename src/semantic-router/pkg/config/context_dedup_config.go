package config

import "strings"

// Context dedup is a route-local context action that removes the later copy
// of prior turns a request carries twice in a row. Identity is exact: this
// package only describes, defaults, and validates the action's configuration.
const (
	// ContextDedupNormalizationExact compares block text byte for byte.
	ContextDedupNormalizationExact = "exact"
	// ContextDedupNormalizationWhitespace collapses runs of white space and
	// trims the ends before comparing. Case and punctuation are never changed.
	ContextDedupNormalizationWhitespace = "whitespace"

	ContextDedupFailureOpen   = "fail_open"
	ContextDedupFailureClosed = "fail_closed"
)

// Planning bounds. Defaults apply when the policy omits a limit; the maxima
// keep per-request preparation work bounded regardless of configuration.
const (
	DefaultContextDedupMaxHistoryTurns = 128
	DefaultContextDedupMaxHistoryBytes = 1 << 20
	DefaultContextDedupMaxSegmentTurns = 64
	DefaultContextDedupTimeoutMs       = 50

	maxContextDedupHistoryTurns = 4096
	maxContextDedupHistoryBytes = 64 << 20
	maxContextDedupSegmentTurns = 2048
	maxContextDedupTimeoutMs    = 5000
)

// ContextDedupPluginConfig controls route-local removal of adjacent repeated
// history turns. Omission or enabled=false performs no deduplication work.
type ContextDedupPluginConfig struct {
	Enabled       bool                      `json:"enabled" yaml:"enabled"`
	Normalization string                    `json:"normalization,omitempty" yaml:"normalization,omitempty"`
	FailureMode   string                    `json:"failure_mode,omitempty" yaml:"failure_mode,omitempty"`
	Limits        *ContextDedupLimitsConfig `json:"limits,omitempty" yaml:"limits,omitempty"`
}

// ContextDedupLimitsConfig bounds the history a single request may examine.
// MaxHistoryBytes counts the text the policy inspects; tool arguments, media,
// and other non-text payloads are not part of that view, so TimeoutMs is the
// bound that covers expensive inputs regardless of their shape.
// MaxSegmentTurns bounds how many consecutive turns one repeated block may
// span; a whole history re-sent as one block needs a bound at least as large
// as that history.
type ContextDedupLimitsConfig struct {
	MaxHistoryTurns int `json:"max_history_turns,omitempty" yaml:"max_history_turns,omitempty"`
	MaxHistoryBytes int `json:"max_history_bytes,omitempty" yaml:"max_history_bytes,omitempty"`
	MaxSegmentTurns int `json:"max_segment_turns,omitempty" yaml:"max_segment_turns,omitempty"`
	TimeoutMs       int `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`
}

// IsEnabled reports whether the decision configured an active policy.
func (c *ContextDedupPluginConfig) IsEnabled() bool {
	return c != nil && c.Enabled
}

func (c *ContextDedupPluginConfig) EffectiveNormalization() string {
	if c == nil || strings.TrimSpace(c.Normalization) == "" {
		return ContextDedupNormalizationExact
	}
	return strings.TrimSpace(c.Normalization)
}

func (c *ContextDedupPluginConfig) EffectiveFailureMode() string {
	if c == nil || strings.TrimSpace(c.FailureMode) == "" {
		return ContextDedupFailureOpen
	}
	return strings.TrimSpace(c.FailureMode)
}

// EffectiveLimits fills each omitted bound with its default. Configured values
// are returned as written; validation rejects nonpositive or excessive ones.
func (c *ContextDedupPluginConfig) EffectiveLimits() ContextDedupLimitsConfig {
	result := ContextDedupLimitsConfig{
		MaxHistoryTurns: DefaultContextDedupMaxHistoryTurns,
		MaxHistoryBytes: DefaultContextDedupMaxHistoryBytes,
		MaxSegmentTurns: DefaultContextDedupMaxSegmentTurns,
		TimeoutMs:       DefaultContextDedupTimeoutMs,
	}
	if c == nil || c.Limits == nil {
		return result
	}
	if c.Limits.MaxHistoryTurns != 0 {
		result.MaxHistoryTurns = c.Limits.MaxHistoryTurns
	}
	if c.Limits.MaxHistoryBytes != 0 {
		result.MaxHistoryBytes = c.Limits.MaxHistoryBytes
	}
	if c.Limits.MaxSegmentTurns != 0 {
		result.MaxSegmentTurns = c.Limits.MaxSegmentTurns
	}
	if c.Limits.TimeoutMs != 0 {
		result.TimeoutMs = c.Limits.TimeoutMs
	}
	return result
}

// GetContextDedupConfig returns route-local context deduplication settings.
func (d *Decision) GetContextDedupConfig() *ContextDedupPluginConfig {
	result := &ContextDedupPluginConfig{}
	return decodeDecisionPlugin(d, DecisionPluginContextDedup, result)
}
