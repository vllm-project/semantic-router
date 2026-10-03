package config

import (
	"strconv"
	"time"
)

// SignalTypeDecision asks a decision model a typed question about the request
// through the built-in model runtime (a model_runtime deployment).
const SignalTypeDecision = "decision"

const (
	DecisionQuestionChoice = "choice"
	DecisionQuestionNoul   = "noul"
	DecisionQuestionScore  = "score"

	// DefaultDecisionTimeoutMs bounds one decision call when a rule sets none.
	DefaultDecisionTimeoutMs = 1000
	MaxDecisionTimeoutMs     = 60000
	MinDecisionChoices       = 2
	MaxDecisionChoices       = 255
	MinDecisionLevels        = 2
	MaxDecisionLevels        = 10
	// DefaultDecisionNoulThreshold is the P(true) a Noul answer needs to match
	// when its rule declares no predicate.
	DefaultDecisionNoulThreshold = 0.5
)

// DecisionSignalRule is one named question to a decision model. Noul answers
// match when P(true) satisfies the predicate (default gte 0.5); Score answers
// match when the expected level satisfies the predicate (required); Choice
// answers are label-qualified: a condition names the option key, and it
// matches the arg-max option, only when its probability satisfies the
// predicate if one is declared. Every probability is also published as a
// signal value, so conditions can apply their own predicate. A late or failed
// answer leaves the signal unknown.
type DecisionSignalRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Deployment  string            `yaml:"deployment"`
	Question    DecisionQuestion  `yaml:"question"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
}

// DecisionQuestion is a System One question with ordered options.
type DecisionQuestion struct {
	Type         string           `yaml:"type"`
	Instructions string           `yaml:"instructions"`
	Choices      []DecisionChoice `yaml:"choices,omitempty"`
	Levels       []string         `yaml:"levels,omitempty"`
}

// DecisionChoice is one Choice option, or the false/true descriptions of a Noul.
type DecisionChoice struct {
	Key         string `yaml:"key"`
	Description string `yaml:"description,omitempty"`
}

// DecisionSelectionConfig asks a decision model to choose one of a routing
// decision's modelRefs. Options are the candidate model names; descriptions
// come from Candidates, then the model's configured description.
type DecisionSelectionConfig struct {
	Deployment   string            `yaml:"deployment"`
	Instructions string            `yaml:"instructions"`
	Candidates   map[string]string `yaml:"candidates,omitempty"`
	TimeoutMs    int               `yaml:"timeout_ms,omitempty"`
}

// EffectiveTimeout is the rule's call deadline.
func (r DecisionSignalRule) EffectiveTimeout() time.Duration {
	return decisionTimeout(r.TimeoutMs)
}

// EffectivePredicate is the predicate a Noul or Score answer must satisfy, or
// the optional predicate on a Choice answer's chosen probability.
func (r DecisionSignalRule) EffectivePredicate() *NumericPredicate {
	if r.Predicate != nil || r.Question.Type != DecisionQuestionNoul {
		return r.Predicate
	}
	threshold := DefaultDecisionNoulThreshold
	return &NumericPredicate{GTE: &threshold}
}

// OptionKeys are the answer keys in rendering order.
func (q DecisionQuestion) OptionKeys() []string {
	switch q.Type {
	case DecisionQuestionScore:
		keys := make([]string, len(q.Levels))
		for index := range q.Levels {
			keys[index] = strconv.Itoa(index)
		}
		return keys
	case DecisionQuestionNoul:
		return []string{"false", "true"}
	default:
		keys := make([]string, len(q.Choices))
		for index, choice := range q.Choices {
			keys[index] = choice.Key
		}
		return keys
	}
}

// EffectiveTimeout is the selector's call deadline.
func (c DecisionSelectionConfig) EffectiveTimeout() time.Duration {
	return decisionTimeout(c.TimeoutMs)
}

func decisionTimeout(milliseconds int) time.Duration {
	if milliseconds <= 0 {
		milliseconds = DefaultDecisionTimeoutMs
	}
	return time.Duration(milliseconds) * time.Millisecond
}
