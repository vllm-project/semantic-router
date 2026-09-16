package config

import "fmt"

const (
	CandidateCapabilitiesDeclared = "declared"
	CandidateContextKnownLimits   = "known_limits"
)

// CandidateRequirements narrows a recipe's assigned models using declared
// capabilities and known limits. Input accounting remains an estimate; this
// policy does not assert exact provider token capacity.
type CandidateRequirements struct {
	Capabilities string `yaml:"capabilities,omitempty" json:"capabilities,omitempty"`
	Context      string `yaml:"context,omitempty" json:"context,omitempty"`
}

func (r *CandidateRequirements) Validate() error {
	if r == nil {
		return nil
	}
	if r.Capabilities != "" && r.Capabilities != CandidateCapabilitiesDeclared {
		return fmt.Errorf("candidate_requirements.capabilities must be %q", CandidateCapabilitiesDeclared)
	}
	if r.Context != "" && r.Context != CandidateContextKnownLimits {
		return fmt.Errorf("candidate_requirements.context must be %q", CandidateContextKnownLimits)
	}
	return nil
}

func (r *CandidateRequirements) Clone() *CandidateRequirements {
	if r == nil {
		return nil
	}
	clone := *r
	return &clone
}
