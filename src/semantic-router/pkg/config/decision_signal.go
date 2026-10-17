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
	// DecisionQuestionSet asks which of the labels apply; models that declare
	// it (Vela 2.0) answer every label with its own probability.
	DecisionQuestionSet = "set"
	// DecisionQuestionSpan asks where in the text each label occurs; models
	// that declare it (Vela 2.0) answer labelled character spans.
	DecisionQuestionSpan = "span"

	// DecisionSpanHeadRouter and DecisionSpanHeadBroad name the span heads of
	// models with two: the router head (PII, unsupported claims, toxic spans)
	// and the broad head (open extraction).
	DecisionSpanHeadRouter = "router"
	DecisionSpanHeadBroad  = "broad"

	// DefaultDecisionTimeoutMs bounds one decision call when a rule sets none.
	DefaultDecisionTimeoutMs = 1000
	MaxDecisionTimeoutMs     = 60000
	// MaxDecisionPriorUserTurns bounds how many earlier user turns a decision
	// question reads before the current one.
	MaxDecisionPriorUserTurns = 8
	MinDecisionChoices        = 2
	MaxDecisionChoices        = 255
	MinDecisionLevels         = 2
	MaxDecisionLevels         = 10
	MinDecisionLabels         = 1
	MaxDecisionLabels         = 255
	// DefaultDecisionNoulThreshold is the P(true) a Noul answer needs to match
	// when its rule declares no predicate.
	DefaultDecisionNoulThreshold = 0.5
)

// DecisionSignalRule is one named question to a decision model. Noul answers
// match when P(true) satisfies the predicate (default gte 0.5); Score answers
// match when the expected level satisfies the predicate (required); Choice
// answers are label-qualified: a condition names the option key, and it
// matches the arg-max option, only when its probability satisfies the
// predicate if one is declared. Set and Span answers are label-qualified too:
// a Set label matches when its probability satisfies the predicate (without
// one, when the model selected it), and a Span label when one of its spans'
// probabilities does (without one, when the model found a span of it). Every
// probability is also published as a signal value, so conditions can apply
// their own predicate. A late or failed answer leaves the signal unknown.
type DecisionSignalRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Deployment  string            `yaml:"deployment,omitempty"`
	Question    DecisionQuestion  `yaml:"question"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
	TimeoutMs   int               `yaml:"timeout_ms,omitempty"`
	// PriorUserTurns puts up to this many of the conversation's earlier user
	// turns, oldest first, before the current one in the question's state,
	// so a follow-up such as "now make it shorter" is read with what it
	// follows.
	PriorUserTurns int `yaml:"prior_user_turns,omitempty"`
}

// DecisionQuestion is a System One question with ordered options. Choice and
// Noul take choices and Score takes levels; Set and Span take labels, an
// optional threshold that replaces the model's own, and, for Span, the head
// that answers.
type DecisionQuestion struct {
	Type         string           `yaml:"type"`
	Instructions string           `yaml:"instructions"`
	Choices      []DecisionChoice `yaml:"choices,omitempty"`
	Levels       []string         `yaml:"levels,omitempty"`
	Labels       []DecisionChoice `yaml:"labels,omitempty"`
	Threshold    *float64         `yaml:"threshold,omitempty"`
	Head         string           `yaml:"head,omitempty"`
}

// DecisionChoice is one Choice option, the false/true descriptions of a Noul,
// or one Set or Span label.
type DecisionChoice struct {
	Key         string `yaml:"key"`
	Description string `yaml:"description,omitempty"`
}

// DecisionSelectionConfig asks a decision model to choose one of a routing
// decision's modelRefs. Options are the candidate model names; descriptions
// come from Candidates, then the model's configured description. Without a
// Deployment the Router's decision model chooses, on the deployment that
// already answers the request's signals.
type DecisionSelectionConfig struct {
	Deployment   string            `yaml:"deployment,omitempty"`
	Instructions string            `yaml:"instructions"`
	Candidates   map[string]string `yaml:"candidates,omitempty"`
	TimeoutMs    int               `yaml:"timeout_ms,omitempty"`
}

// EffectiveTimeout is the rule's call deadline.
func (r DecisionSignalRule) EffectiveTimeout() time.Duration {
	return decisionTimeout(r.TimeoutMs)
}

// EffectivePredicate is the predicate a Noul or Score answer must satisfy, or
// the optional predicate on a Choice option's or a Set or Span label's
// probability.
func (r DecisionSignalRule) EffectivePredicate() *NumericPredicate {
	if r.Predicate != nil || r.Question.Type != DecisionQuestionNoul {
		return r.Predicate
	}
	threshold := DefaultDecisionNoulThreshold
	return &NumericPredicate{GTE: &threshold}
}

// Labelled reports whether conditions on the question name one of its
// options: a Choice option or a Set or Span label.
func (q DecisionQuestion) Labelled() bool {
	switch q.Type {
	case DecisionQuestionChoice, DecisionQuestionSet, DecisionQuestionSpan:
		return true
	}
	return false
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
	case DecisionQuestionSet, DecisionQuestionSpan:
		return decisionChoiceKeys(q.Labels)
	default:
		return decisionChoiceKeys(q.Choices)
	}
}

func decisionChoiceKeys(choices []DecisionChoice) []string {
	keys := make([]string, len(choices))
	for index, choice := range choices {
		keys[index] = choice.Key
	}
	return keys
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
