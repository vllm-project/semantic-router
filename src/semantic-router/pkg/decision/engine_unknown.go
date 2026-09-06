package decision

import (
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var ErrDecisionUnresolved = errors.New("decision unresolved")

type DecisionUnresolvedError struct {
	Decision string
}

func (e *DecisionUnresolvedError) Error() string {
	return fmt.Sprintf("decision %q could not be resolved because a signal evaluator failed: %v", e.Decision, ErrDecisionUnresolved)
}

func (e *DecisionUnresolvedError) Unwrap() error {
	return ErrDecisionUnresolved
}

func applyUnknownPolicy(
	decision *config.Decision,
	evaluation nodeEvaluation,
	policy config.UnknownPolicy,
) (nodeEvaluation, error) {
	switch policy {
	case config.RuleOnUnknownMatch:
		evaluation.state = evaluationTrue
		evaluation.confidence = 1
		evaluation.scored = false
		evaluation.matchedRules = []string{"on_unknown:match"}
		return evaluation, nil
	case config.RuleOnUnknownNoMatch:
		evaluation.state = evaluationFalse
		evaluation.confidence = 0
		evaluation.scored = false
		return evaluation, nil
	case config.RuleOnUnknownFailRequest:
		return evaluation, &DecisionUnresolvedError{Decision: decision.Name}
	default:
		return evaluation, fmt.Errorf("decision %q has invalid on_unknown policy %q", decision.Name, policy)
	}
}
