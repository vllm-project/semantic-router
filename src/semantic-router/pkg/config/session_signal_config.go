package config

import "strings"

const (
	SessionFactSessionPresent   = "session_present"
	SessionFactHasPreviousModel = "has_previous_model"
	SessionFactTurnIndex        = "turn_index"
	SessionFactCacheWarmth      = "cache_warmth"
	SessionFactRemainingTurns   = "remaining_turns"
	SessionFactHandoffPenalty   = "handoff_penalty"
	SessionFactQualityGap       = "quality_gap"
)

// SessionRule declares a runtime-derived session signal that can be referenced
// directly from routing decisions and projection inputs.
type SessionRule struct {
	Name           string            `yaml:"name"`
	Description    string            `yaml:"description,omitempty"`
	Fact           string            `yaml:"fact"`
	Predicate      *NumericPredicate `yaml:"predicate,omitempty"`
	IntentOrDomain string            `yaml:"intent_or_domain,omitempty"`
	PreviousModel  string            `yaml:"previous_model,omitempty"`
	CandidateModel string            `yaml:"candidate_model,omitempty"`
}

func NormalizeSessionFact(fact string) string {
	return strings.ToLower(strings.TrimSpace(fact))
}

func IsSupportedSessionFact(fact string) bool {
	switch NormalizeSessionFact(fact) {
	case SessionFactSessionPresent,
		SessionFactHasPreviousModel,
		SessionFactTurnIndex,
		SessionFactCacheWarmth,
		SessionFactRemainingTurns,
		SessionFactHandoffPenalty,
		SessionFactQualityGap:
		return true
	default:
		return false
	}
}
