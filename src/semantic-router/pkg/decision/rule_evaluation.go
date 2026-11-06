package decision

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RuleState is the three-valued outcome of one rule tree.
type RuleState uint8

const (
	RuleFalse RuleState = iota
	RuleTrue
	RuleUnknown
)

func (s RuleState) String() string {
	switch s {
	case RuleTrue:
		return evaluationTrue.String()
	case RuleUnknown:
		return evaluationUnknown.String()
	default:
		return evaluationFalse.String()
	}
}

// RuleOutcome is what one rule tree read: its state and the conditions that
// matched.
type RuleOutcome struct {
	State        RuleState
	MatchedRules []string
}

// keepUnknown is the policy EvaluateRule reads with. Under any policy a
// condition whose signal failed reads as unknown, not through its on_error;
// EvaluateRule's caller resolves an unknown outcome itself.
const keepUnknown = config.RuleOnUnknownNoMatch

// EvaluateRule reads one rule tree outside decision selection, such as a
// response_guard rule's when, against signals. A condition whose signal
// failed is unknown, and an AND stops at its first false condition. Before
// it reads a condition it calls observe, when set, so a signal can be
// evaluated only once a tree reads it.
func (e *DecisionEngine) EvaluateRule(node config.RuleNode, signals *SignalMatches, observe func(config.RuleNode)) RuleOutcome {
	reader := *e
	reader.observe = observe
	evaluation, _ := reader.evalNode(node, signals, keepUnknown, false)
	outcome := RuleOutcome{State: RuleFalse, MatchedRules: evaluation.matchedRules}
	switch evaluation.state {
	case evaluationTrue:
		outcome.State = RuleTrue
	case evaluationUnknown:
		outcome.State = RuleUnknown
	}
	return outcome
}
