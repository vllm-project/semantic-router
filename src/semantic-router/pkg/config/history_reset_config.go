package config

import "strings"

// History reset is a route-local context action that removes complete eligible
// prior turns after an accepted topic change. Detection is not part of this
// contract: the topic-continuity signal owns it, and this package only
// describes, defaults, and validates the action's configuration.
const (
	// HistoryResetScopeEligibleHistory is the only supported reset scope.
	// Ranking or selective retention belong to separate context actions.
	HistoryResetScopeEligibleHistory = "eligible_history"

	HistoryResetFailureOpen   = "fail_open"
	HistoryResetFailureClosed = "fail_closed"

	// HistoryResetTriggerSignalType is the only signal family allowed to
	// authorize removal. An enabled policy whose family is unregistered is
	// rejected rather than allowed to bind to an unrelated signal that
	// happens to resolve.
	HistoryResetTriggerSignalType = "topic_continuity"

	// HistoryResetTriggerUnavailable is the stable reason reported when an
	// enabled policy cannot resolve a usable topic-continuity trigger.
	HistoryResetTriggerUnavailable = "history_reset_trigger_unavailable"
)

// Planning bounds. Defaults apply when the policy omits a limit; the maxima
// keep per-request preparation work bounded regardless of configuration.
const (
	DefaultHistoryResetMaxHistoryTurns = 128
	DefaultHistoryResetMaxHistoryBytes = 1 << 20
	DefaultHistoryResetTimeoutMs       = 50

	maxHistoryResetHistoryTurns = 4096
	maxHistoryResetHistoryBytes = 64 << 20
	maxHistoryResetTimeoutMs    = 5000
)

// HistoryResetRecoveryConfig reuses the shared context-recovery shape so reset
// and compression resolve to one request-level store, budget, and retrieval
// tool instead of two independently configured recovery paths.
type HistoryResetRecoveryConfig = ContextCompressionRecoveryConfig

// HistoryResetPluginConfig controls route-local removal of eligible prior
// history. Omission or enabled=false performs no reset work.
type HistoryResetPluginConfig struct {
	Enabled     bool                        `json:"enabled" yaml:"enabled"`
	Trigger     *HistoryResetTriggerConfig  `json:"trigger,omitempty" yaml:"trigger,omitempty"`
	Scope       string                      `json:"scope,omitempty" yaml:"scope,omitempty"`
	FailureMode string                      `json:"failure_mode,omitempty" yaml:"failure_mode,omitempty"`
	Limits      *HistoryResetLimitsConfig   `json:"limits,omitempty" yaml:"limits,omitempty"`
	Recovery    *HistoryResetRecoveryConfig `json:"recovery,omitempty" yaml:"recovery,omitempty"`
}

// HistoryResetTriggerConfig binds the action to one topic-continuity signal.
// Client headers and message content cannot authorize a reset.
type HistoryResetTriggerConfig struct {
	Signal        string   `json:"signal,omitempty" yaml:"signal,omitempty"`
	MinConfidence *float64 `json:"min_confidence,omitempty" yaml:"min_confidence,omitempty"`
	// AcceptedVersions pins the producing signal contracts this policy trusts.
	// An enabled policy must name at least one: without it the action would
	// have to trust whatever version a producer claims.
	AcceptedVersions []string `json:"accepted_versions,omitempty" yaml:"accepted_versions,omitempty"`
}

// HistoryResetLimitsConfig bounds the history a single request may examine.
// MaxHistoryBytes counts the text the policy inspects; tool arguments, media,
// and other non-text payloads are not part of that view, so TimeoutMs is the
// bound that covers expensive inputs regardless of their shape.
type HistoryResetLimitsConfig struct {
	MaxHistoryTurns int `json:"max_history_turns,omitempty" yaml:"max_history_turns,omitempty"`
	MaxHistoryBytes int `json:"max_history_bytes,omitempty" yaml:"max_history_bytes,omitempty"`
	TimeoutMs       int `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`
}

// IsEnabled reports whether the decision configured an active reset policy.
func (c *HistoryResetPluginConfig) IsEnabled() bool {
	return c != nil && c.Enabled
}

func (c *HistoryResetPluginConfig) EffectiveScope() string {
	if c == nil || strings.TrimSpace(c.Scope) == "" {
		return HistoryResetScopeEligibleHistory
	}
	return strings.TrimSpace(c.Scope)
}

func (c *HistoryResetPluginConfig) EffectiveFailureMode() string {
	if c == nil || strings.TrimSpace(c.FailureMode) == "" {
		return HistoryResetFailureOpen
	}
	return strings.TrimSpace(c.FailureMode)
}

// EffectiveLimits fills each omitted bound with its default. Configured values
// are returned as written; validation rejects nonpositive or excessive ones.
func (c *HistoryResetPluginConfig) EffectiveLimits() HistoryResetLimitsConfig {
	result := HistoryResetLimitsConfig{
		MaxHistoryTurns: DefaultHistoryResetMaxHistoryTurns,
		MaxHistoryBytes: DefaultHistoryResetMaxHistoryBytes,
		TimeoutMs:       DefaultHistoryResetTimeoutMs,
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
	if c.Limits.TimeoutMs != 0 {
		result.TimeoutMs = c.Limits.TimeoutMs
	}
	return result
}

// EffectiveMinConfidence returns the configured acceptance threshold. The
// second result reports whether the policy supplied one; validation requires
// an explicit threshold before a policy may be enabled.
func (c *HistoryResetPluginConfig) EffectiveMinConfidence() (float64, bool) {
	if c == nil || c.Trigger == nil || c.Trigger.MinConfidence == nil {
		return 0, false
	}
	return *c.Trigger.MinConfidence, true
}

// RequiresRecovery reports whether removal may only proceed once the removed
// turns have been stored recoverably.
func (c *HistoryResetPluginConfig) RequiresRecovery() bool {
	return c != nil && c.Recovery != nil && c.Recovery.Enabled
}

// historyResetTriggerFamilyRegistered reports whether a signal catalog carries
// the topic-continuity family. The gate opens on its own when the family is
// registered; no build flag or manual switch is involved.
func historyResetTriggerFamilyRegistered(catalog []SignalCatalogEntry) bool {
	for _, entry := range catalog {
		if entry.Type == HistoryResetTriggerSignalType {
			return true
		}
	}
	return false
}

// GetHistoryResetConfig returns route-local history-reset settings.
func (d *Decision) GetHistoryResetConfig() *HistoryResetPluginConfig {
	result := &HistoryResetPluginConfig{}
	return decodeDecisionPlugin(d, DecisionPluginHistoryReset, result)
}
