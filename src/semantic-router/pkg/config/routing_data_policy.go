package config

// RoutingDataPolicy places standing data-use limits on a recipe, including
// requests that fail before selecting a decision. Limits can only tighten
// the global and per-decision policies for their respective surfaces.
type RoutingDataPolicy struct {
	// Replay false forbids capture. Unset or true preserves existing replay
	// enablement and never enables capture by itself.
	Replay *bool `yaml:"replay,omitempty" json:"replay,omitempty"`
}

func (p *RoutingDataPolicy) ReplayAllowed() bool {
	return p == nil || p.Replay == nil || *p.Replay
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
	return &cloned
}
