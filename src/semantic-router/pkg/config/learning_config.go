package config

import "strings"

const (
	// RouterLearningScopeConversation protects one agent conversation while
	// allowing later conversations in the same session to be re-evaluated.
	RouterLearningScopeConversation = "conversation"
	// RouterLearningScopeSession protects the whole declared agent session.
	RouterLearningScopeSession = "session"

	// DecisionAdaptationModeApply lets learning affect the final selected model.
	DecisionAdaptationModeApply = "apply"
	// DecisionAdaptationModeObserve computes diagnostics but leaves selection unchanged.
	DecisionAdaptationModeObserve = "observe"
	// DecisionAdaptationModeBypass disables learning for a hard policy route.
	DecisionAdaptationModeBypass = "bypass"

	// RouterLearningCandidateSetDecision lets adaptation search within the matched decision.
	RouterLearningCandidateSetDecision = "decision"
	// RouterLearningCandidateSetTier lets adaptation search the matched decision tier.
	RouterLearningCandidateSetTier = "tier"
	// RouterLearningCandidateSetGlobal lets adaptation search the deployed model inventory.
	RouterLearningCandidateSetGlobal = "global"

	// RouterLearningStrategyRoutingSampling is the default online model-choice algorithm.
	RouterLearningStrategyRoutingSampling = "routing_sampling"
)

const (
	routerLearningDefaultSessionHeader      = "x-session-id"
	routerLearningDefaultConversationHeader = "x-conversation-id"
)

// RouterLearningConfig is the public cross-request routing intelligence
// surface. Adaptation proposes model-choice improvements; protection controls
// exploration and model switches for agent continuity.
type RouterLearningConfig struct {
	Enabled    bool                           `yaml:"enabled,omitempty"`
	Adaptation RouterLearningAdaptationConfig `yaml:"adaptation,omitempty"`
	Protection RouterLearningProtectionConfig `yaml:"protection,omitempty"`
}

// RouterLearningAdaptationConfig configures online model-choice learning.
type RouterLearningAdaptationConfig struct {
	Enabled      *bool  `yaml:"enabled,omitempty"`
	CandidateSet string `yaml:"candidate_set,omitempty"`
	Strategy     string `yaml:"strategy,omitempty"`
}

// RouterLearningProtectionConfig configures online stability protection.
type RouterLearningProtectionConfig struct {
	Enabled  *bool                          `yaml:"enabled,omitempty"`
	Scope    string                         `yaml:"scope,omitempty"`
	Identity RouterLearningIdentityConfig   `yaml:"identity,omitempty"`
	Tuning   RouterLearningProtectionTuning `yaml:"tuning,omitempty"`
}

// RouterLearningIdentityConfig declares where protection reads stable client
// identity. The runtime currently uses the session and conversation keys.
type RouterLearningIdentityConfig struct {
	Headers RouterLearningIdentityHeadersConfig `yaml:"headers,omitempty"`
}

// RouterLearningIdentityHeadersConfig is intentionally typed. The public
// protection identity shape supports only the headers used by the protection
// runtime; future identity needs should add typed fields instead of accepting
// arbitrary header keys.
type RouterLearningIdentityHeadersConfig struct {
	Session      *string `yaml:"session,omitempty"`
	Conversation *string `yaml:"conversation,omitempty"`
}

// RouterLearningProtectionTuning exposes the stable protection knobs.
type RouterLearningProtectionTuning struct {
	IdleTimeoutSeconds   *int     `yaml:"idle_timeout_seconds,omitempty"`
	MinTurnsBeforeSwitch *int     `yaml:"min_turns_before_switch,omitempty"`
	SwitchMargin         *float64 `yaml:"switch_margin,omitempty"`
	StabilityWeight      *float64 `yaml:"stability_weight,omitempty"`
}

// DecisionAdaptationsConfig lets one matched decision control globally enabled
// learning. The name stays plural because it controls multiple learning
// components at the decision boundary.
type DecisionAdaptationsConfig struct {
	Mode       string                            `yaml:"mode,omitempty"`
	Adaptation *DecisionLearningAdaptationConfig `yaml:"adaptation,omitempty"`
	Protection *DecisionLearningProtectionConfig `yaml:"protection,omitempty"`
}

// DecisionLearningAdaptationConfig controls decision-local model-choice
// learning. Empty mode and candidate_set inherit the global learning defaults.
type DecisionLearningAdaptationConfig struct {
	Mode         string `yaml:"mode,omitempty"`
	CandidateSet string `yaml:"candidate_set,omitempty"`
}

// DecisionLearningProtectionConfig controls decision-local stability
// protection. Empty mode inherits apply.
type DecisionLearningProtectionConfig struct {
	Mode            string   `yaml:"mode,omitempty"`
	StabilityWeight *float64 `yaml:"stability_weight,omitempty"`
	SwitchMargin    *float64 `yaml:"switch_margin,omitempty"`
}

func (cfg RouterLearningAdaptationConfig) EffectiveCandidateSet() string {
	if cfg.CandidateSet == "" {
		return RouterLearningCandidateSetDecision
	}
	return cfg.CandidateSet
}

func (cfg RouterLearningAdaptationConfig) EffectiveEnabled() bool {
	return cfg.Enabled == nil || *cfg.Enabled
}

func (cfg RouterLearningAdaptationConfig) EffectiveStrategy() string {
	if cfg.Strategy == "" {
		return RouterLearningStrategyRoutingSampling
	}
	return cfg.Strategy
}

func (cfg RouterLearningProtectionConfig) EffectiveEnabled() bool {
	return cfg.Enabled == nil || *cfg.Enabled
}

func (cfg RouterLearningProtectionConfig) EffectiveScope() string {
	if cfg.Scope == "" {
		return RouterLearningScopeConversation
	}
	return cfg.Scope
}

func (cfg RouterLearningProtectionConfig) HeaderName(key string) string {
	switch key {
	case "session":
		if cfg.Identity.Headers.Session != nil {
			if value := strings.TrimSpace(*cfg.Identity.Headers.Session); value != "" {
				return value
			}
		}
		return routerLearningDefaultSessionHeader
	case "conversation":
		if cfg.Identity.Headers.Conversation != nil {
			if value := strings.TrimSpace(*cfg.Identity.Headers.Conversation); value != "" {
				return value
			}
		}
		return routerLearningDefaultConversationHeader
	default:
		return ""
	}
}

func (cfg DecisionAdaptationsConfig) EffectiveMode() string {
	if cfg.Mode == "" {
		return DecisionAdaptationModeApply
	}
	return cfg.Mode
}

func (cfg DecisionAdaptationsConfig) AdaptationMode() string {
	decisionMode := cfg.EffectiveMode()
	if decisionMode == DecisionAdaptationModeBypass {
		return decisionMode
	}
	if cfg.Adaptation == nil || cfg.Adaptation.Mode == "" {
		return decisionMode
	}
	if decisionMode == DecisionAdaptationModeObserve && cfg.Adaptation.Mode == DecisionAdaptationModeApply {
		return DecisionAdaptationModeObserve
	}
	return cfg.Adaptation.Mode
}

func (cfg DecisionAdaptationsConfig) AdaptationCandidateSet(globalCandidateSet string) string {
	if cfg.Adaptation != nil && cfg.Adaptation.CandidateSet != "" {
		return cfg.Adaptation.CandidateSet
	}
	if globalCandidateSet == "" {
		return RouterLearningCandidateSetDecision
	}
	return globalCandidateSet
}

func (cfg DecisionAdaptationsConfig) ProtectionMode() string {
	decisionMode := cfg.EffectiveMode()
	if decisionMode == DecisionAdaptationModeBypass {
		return decisionMode
	}
	if cfg.Protection == nil || cfg.Protection.Mode == "" {
		return decisionMode
	}
	if decisionMode == DecisionAdaptationModeObserve && cfg.Protection.Mode == DecisionAdaptationModeApply {
		return DecisionAdaptationModeObserve
	}
	return cfg.Protection.Mode
}

func (cfg DecisionAdaptationsConfig) ApplyProtectionTuning(
	tuning RouterLearningProtectionTuning,
) RouterLearningProtectionTuning {
	if cfg.Protection == nil {
		return tuning
	}
	if cfg.Protection.StabilityWeight != nil {
		tuning.StabilityWeight = cfg.Protection.StabilityWeight
	}
	if cfg.Protection.SwitchMargin != nil {
		tuning.SwitchMargin = cfg.Protection.SwitchMargin
	}
	return tuning
}
