package config

const (
	// RouterLearningScopeConversation protects one agent conversation while
	// allowing later conversations in the same session to be re-evaluated.
	RouterLearningScopeConversation = "conversation"
	// RouterLearningScopeSession protects the whole declared agent session.
	RouterLearningScopeSession = "session"

	// DecisionAdaptationModeApply lets an adaptation change the selected model.
	DecisionAdaptationModeApply = "apply"
	// DecisionAdaptationModeObserve computes diagnostics but leaves selection unchanged.
	DecisionAdaptationModeObserve = "observe"
	// DecisionAdaptationModeBypass disables an adaptation for a hard policy route.
	DecisionAdaptationModeBypass = "bypass"
)

// RouterLearningConfig is the public cross-request routing intelligence surface.
// It owns adaptation registration while concrete storage controls remain with
// existing services such as global.services.router_replay.
type RouterLearningConfig struct {
	Enabled     bool                      `yaml:"enabled,omitempty"`
	Adaptations RouterLearningAdaptations `yaml:"adaptations,omitempty"`
}

// RouterLearningAdaptations contains globally registered learning adaptations.
type RouterLearningAdaptations struct {
	SessionAware SessionAwareLearningConfig `yaml:"session_aware,omitempty"`
}

// SessionAwareLearningConfig configures session-aware Router Learning.
type SessionAwareLearningConfig struct {
	Enabled  bool                       `yaml:"enabled,omitempty"`
	Scope    string                     `yaml:"scope,omitempty"`
	Identity SessionAwareIdentityConfig `yaml:"identity,omitempty"`
	Tuning   SessionAwareLearningTuning `yaml:"tuning,omitempty"`
}

// SessionAwareIdentityConfig declares where session-aware learning reads
// stable client identity. Unknown header keys are preserved for future
// adaptations but the session-aware runtime currently uses session and
// conversation.
type SessionAwareIdentityConfig struct {
	Headers map[string]string `yaml:"headers,omitempty"`
}

// SessionAwareLearningTuning exposes the stable session-aware knobs that remain
// meaningful after moving the feature out of decision.algorithm.
type SessionAwareLearningTuning struct {
	IdleTimeoutSeconds     *int     `yaml:"idle_timeout_seconds,omitempty"`
	MinTurnsBeforeSwitch   *int     `yaml:"min_turns_before_switch,omitempty"`
	SwitchMargin           *float64 `yaml:"switch_margin,omitempty"`
	CacheWeight            *float64 `yaml:"cache_weight,omitempty"`
	HandoffPenalty         *float64 `yaml:"handoff_penalty,omitempty"`
	HandoffPenaltyWeight   *float64 `yaml:"handoff_penalty_weight,omitempty"`
	SwitchHistoryWeight    *float64 `yaml:"switch_history_weight,omitempty"`
	MaxCacheCostMultiplier *float64 `yaml:"max_cache_cost_multiplier,omitempty"`
}

// DecisionAdaptationsConfig lets one matched decision control globally enabled
// learning adaptations.
type DecisionAdaptationsConfig struct {
	SessionAware *DecisionSessionAwareAdaptationConfig `yaml:"session_aware,omitempty"`
}

// DecisionSessionAwareAdaptationConfig controls the session-aware adaptation
// for one decision. Empty mode inherits the default apply behavior.
type DecisionSessionAwareAdaptationConfig struct {
	Mode   string                     `yaml:"mode,omitempty"`
	Scope  string                     `yaml:"scope,omitempty"`
	Tuning SessionAwareLearningTuning `yaml:"tuning,omitempty"`
}

func (cfg SessionAwareLearningConfig) EffectiveScope() string {
	if cfg.Scope == "" {
		return RouterLearningScopeConversation
	}
	return cfg.Scope
}

func (cfg SessionAwareLearningConfig) HeaderName(key string) string {
	if cfg.Identity.Headers != nil && cfg.Identity.Headers[key] != "" {
		return cfg.Identity.Headers[key]
	}
	switch key {
	case "session":
		return "x-session-id"
	case "conversation":
		return "x-conversation-id"
	default:
		return ""
	}
}

func (cfg DecisionAdaptationsConfig) SessionAwareMode() string {
	if cfg.SessionAware == nil || cfg.SessionAware.Mode == "" {
		return DecisionAdaptationModeApply
	}
	return cfg.SessionAware.Mode
}
