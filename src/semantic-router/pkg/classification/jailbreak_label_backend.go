package classification

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// LabelDecisionBackend returns a model's categorical label with optional real
// score metadata. It does not promise a probability distribution.
type LabelDecisionBackend interface {
	Decide(context.Context, string) (tasks.LabelDecision, error)
}

// Unwrap admission without mistaking every admitted sequence classifier for a
// categorical backend. Admission still encloses the actual Decide invocation.
func jailbreakDecisionBackend(backend SequenceClassifierBackend) LabelDecisionBackend {
	switch typed := backend.(type) {
	case admittedSequenceClassifier:
		if decision := jailbreakDecisionBackend(typed.backend); decision != nil {
			return admittedLabelDecision{decision, typed}
		}
	case *admittedSequenceClassifier:
		if decision := jailbreakDecisionBackend(typed.backend); decision != nil {
			return admittedLabelDecision{decision, *typed}
		}
	default:
		decision, _ := backend.(LabelDecisionBackend)
		return decision
	}
	return nil
}

type admittedLabelDecision struct {
	backend   LabelDecisionBackend
	admission admittedSequenceClassifier
}

func (a admittedLabelDecision) Decide(ctx context.Context, text string) (tasks.LabelDecision, error) {
	return admitModelInference(ctx, a.admission.gate, a.admission.deployment, func() (tasks.LabelDecision, error) { return a.backend.Decide(ctx, text) })
}
