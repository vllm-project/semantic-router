package config

const (
	// RouterLearningScopeDecision scopes learning state to one matched decision.
	RouterLearningScopeDecision = "decision"
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

	// RouterLearningMethodSessionAware is the session/conversation continuity adaptation.
	RouterLearningMethodSessionAware = "session_aware"
	// RouterLearningMethodBandit is the online exploration/exploitation adaptation.
	RouterLearningMethodBandit = "bandit"
	// RouterLearningMethodElo is the pairwise feedback rating adaptation.
	RouterLearningMethodElo = "elo"
	// RouterLearningMethodPersonalization is the user/workspace preference adaptation.
	RouterLearningMethodPersonalization = "personalization"

	// RouterLearningBanditAlgorithmLinUCB is the default day-0 contextual bandit.
	RouterLearningBanditAlgorithmLinUCB = "linucb"
	// RouterLearningBanditAlgorithmLinearThompson preserves the Thompson-style migration path.
	RouterLearningBanditAlgorithmLinearThompson = "linear_thompson"
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
	SessionAware    SessionAwareLearningConfig    `yaml:"session_aware,omitempty"`
	Bandit          BanditLearningConfig          `yaml:"bandit,omitempty"`
	Elo             EloLearningConfig             `yaml:"elo,omitempty"`
	Personalization PersonalizationLearningConfig `yaml:"personalization,omitempty"`
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

// BanditLearningConfig configures the Router Learning bandit adaptation.
type BanditLearningConfig struct {
	Enabled   bool                 `yaml:"enabled,omitempty"`
	Algorithm string               `yaml:"algorithm,omitempty"`
	Scope     string               `yaml:"scope,omitempty"`
	Goals     map[string]float64   `yaml:"goals,omitempty"`
	Tuning    BanditLearningTuning `yaml:"tuning,omitempty"`
}

// BanditLearningTuning exposes sparse day-0 bandit controls.
type BanditLearningTuning struct {
	ExplorationBudget *float64 `yaml:"exploration_budget,omitempty"`
}

// EloLearningConfig configures the Router Learning Elo adaptation.
type EloLearningConfig struct {
	Enabled       bool     `yaml:"enabled,omitempty"`
	Scope         string   `yaml:"scope,omitempty"`
	InitialRating *float64 `yaml:"initial_rating,omitempty"`
	KFactor       *float64 `yaml:"k_factor,omitempty"`
}

// PersonalizationLearningConfig configures the personalization adaptation.
type PersonalizationLearningConfig struct {
	Enabled bool   `yaml:"enabled,omitempty"`
	Scope   string `yaml:"scope,omitempty"`
}

// DecisionAdaptationsConfig lets one matched decision control globally enabled
// learning adaptations.
type DecisionAdaptationsConfig struct {
	SessionAware    *DecisionSessionAwareAdaptationConfig `yaml:"session_aware,omitempty"`
	Bandit          *DecisionBanditAdaptationConfig       `yaml:"bandit,omitempty"`
	Elo             *DecisionLearningAdaptationConfig     `yaml:"elo,omitempty"`
	Personalization *DecisionLearningAdaptationConfig     `yaml:"personalization,omitempty"`
}

// DecisionSessionAwareAdaptationConfig controls the session-aware adaptation
// for one decision. Empty mode inherits the default apply behavior.
type DecisionSessionAwareAdaptationConfig struct {
	Mode   string                     `yaml:"mode,omitempty"`
	Scope  string                     `yaml:"scope,omitempty"`
	Tuning SessionAwareLearningTuning `yaml:"tuning,omitempty"`
}

// DecisionLearningAdaptationConfig controls a non-session-aware adaptation for
// one decision. Empty mode inherits apply.
type DecisionLearningAdaptationConfig struct {
	Mode string `yaml:"mode,omitempty"`
}

// DecisionBanditAdaptationConfig adds sparse per-decision bandit overrides.
// The global bandit algorithm remains global so state and replay stay coherent.
type DecisionBanditAdaptationConfig struct {
	Mode   string               `yaml:"mode,omitempty"`
	Scope  string               `yaml:"scope,omitempty"`
	Goals  map[string]float64   `yaml:"goals,omitempty"`
	Tuning BanditLearningTuning `yaml:"tuning,omitempty"`
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

func (cfg BanditLearningConfig) EffectiveAlgorithm() string {
	if cfg.Algorithm == "" {
		return RouterLearningBanditAlgorithmLinUCB
	}
	return cfg.Algorithm
}

func (cfg BanditLearningConfig) EffectiveScope() string {
	if cfg.Scope == "" {
		return RouterLearningScopeDecision
	}
	return cfg.Scope
}

func (cfg DecisionAdaptationsConfig) BanditMode() string {
	if cfg.Bandit == nil || cfg.Bandit.Mode == "" {
		return DecisionAdaptationModeApply
	}
	return cfg.Bandit.Mode
}

func (cfg EloLearningConfig) EffectiveScope() string {
	if cfg.Scope == "" {
		return RouterLearningScopeDecision
	}
	return cfg.Scope
}

func (cfg DecisionAdaptationsConfig) EloMode() string {
	if cfg.Elo == nil || cfg.Elo.Mode == "" {
		return DecisionAdaptationModeApply
	}
	return cfg.Elo.Mode
}

func (cfg PersonalizationLearningConfig) EffectiveScope() string {
	if cfg.Scope == "" {
		return RouterLearningScopeDecision
	}
	return cfg.Scope
}

func (cfg DecisionAdaptationsConfig) PersonalizationMode() string {
	if cfg.Personalization == nil || cfg.Personalization.Mode == "" {
		return DecisionAdaptationModeApply
	}
	return cfg.Personalization.Mode
}
