package classification

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// SequenceClassificationResult is the engine-neutral complete distribution in
// a prepared binding's label order. Consumers derive argmax or positive-label
// risk from this distribution rather than synthesizing it from a selected ID.
type SequenceClassificationResult = tasks.LabelDistribution

// SequenceClassifierBackend reports every label's probability, preserving the
// caller's cancellation and deadline across native and remote adapters.
type SequenceClassifierBackend interface {
	Classify(ctx context.Context, text string) (SequenceClassificationResult, error)
}

// CategoryInference retains the historical top-1 and full-distribution methods.
// A provider lacking a full distribution returns it absent, never fabricated.
type CategoryInference interface {
	Classify(ctx context.Context, text string) (tasks.ClassResult, error)
	ClassifyWithProbabilities(ctx context.Context, text string) (tasks.ClassResultWithProbs, error)
}

// PIIInference returns byte-aligned spans in the exact input and an explicit
// error for partial input. PII policy is applied outside the model adapter.
type PIIInference = TokenClassifierBackend

// ScoringBackend reports a raw continuous score. The prepared task/binding
// supplies units, direction and boundaries; this value is not a probability
// unless the model's contract explicitly says so.
type ScoringBackend interface {
	Score(ctx context.Context, text string) (float64, error)
}

// TokenClassifierBackend reports spans with UTF-8 byte offsets into the exact
// input. Wire formats using code points convert once at the adapter boundary.
// Entity labels and outside labels belong to the task, not the engine or PII.
type TokenClassifierBackend interface {
	ClassifyTokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error)
}
