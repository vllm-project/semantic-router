package config

import "fmt"

// RoutingDataPolicy places standing data-use limits on a recipe, including
// requests that fail before selecting a decision. Limits can only tighten
// the global and per-decision policies for their respective surfaces.
type RoutingDataPolicy struct {
	// Replay false forbids capture. Unset or true preserves existing replay
	// enablement and never enables capture by itself.
	Replay *bool `yaml:"replay,omitempty" json:"replay,omitempty"`
	// ReplayPersonalData false keeps the replay record of a request in which
	// a PII signal matched, but without its content: no request or response
	// body, prompt, tool definitions or tool trace. The recipe's PII signals
	// are then evaluated for every request.
	ReplayPersonalData *bool `yaml:"replay_personal_data,omitempty" json:"replay_personal_data,omitempty"`
}

func (p *RoutingDataPolicy) ReplayAllowed() bool {
	return p == nil || p.Replay == nil || *p.Replay
}

// ReplayPersonalDataAllowed reports whether replay may keep the content of a
// request in which a PII signal matched.
func (p *RoutingDataPolicy) ReplayPersonalDataAllowed() bool {
	return p == nil || p.ReplayPersonalData == nil || *p.ReplayPersonalData
}

// Clone preserves unset versus explicit values without sharing mutable flags.
func (p *RoutingDataPolicy) Clone() *RoutingDataPolicy {
	if p == nil {
		return nil
	}
	cloned := *p
	if p.Replay != nil {
		replay := *p.Replay
		cloned.Replay = &replay
	}
	if p.ReplayPersonalData != nil {
		personal := *p.ReplayPersonalData
		cloned.ReplayPersonalData = &personal
	}
	return &cloned
}

// asksPIIForDataPolicy reports whether a routing profile's data policy needs
// its PII signals on every request.
func asksPIIForDataPolicy(policy *RoutingDataPolicy, rules []PIIRule) bool {
	return !policy.ReplayPersonalDataAllowed() && len(rules) > 0
}

// validateRoutingDataPolicy rejects a personal-data replay limit that no PII
// signal could ever trigger.
func validateRoutingDataPolicy(cfg *RouterConfig) error {
	if cfg == nil || cfg.DataPolicy.ReplayPersonalDataAllowed() || len(cfg.PIIRules) > 0 {
		return nil
	}
	return fmt.Errorf("routing.data_policy.replay_personal_data: false needs a routing.signals.pii rule to detect personal data")
}
