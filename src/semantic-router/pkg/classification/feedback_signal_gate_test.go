package classification

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type feedbackCountingGate struct{ calls atomic.Int32 }

func (g *feedbackCountingGate) Acquire(context.Context) (admission.Ticket, error) {
	g.calls.Add(1)
	return nil, admission.ErrQueueFull
}

func TestFeedbackDispatcherDoesNotRunModelOutsideFollowup(t *testing.T) {
	for _, applicable := range []bool{false, true} {
		gate := &feedbackCountingGate{}
		c := &Classifier{
			Config:           &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{UserFeedbackRules: []config.UserFeedbackRule{{Name: "wrong_answer"}}}}},
			feedbackDetector: &FeedbackDetector{initialized: true, gate: gate},
		}
		results := c.EvaluateAllSignalsWithContext("Explain this subject.", "context", "Explain this subject.",
			nil, nil, applicable, true, "", nil, ConversationFacts{}, "")
		if applicable {
			if gate.calls.Load() != 1 || len(results.SignalErrors) != 1 {
				t.Fatal("follow-up did not reach the model admission gate")
			}
		} else if gate.calls.Load() != 0 || len(results.SignalErrors) != 0 || len(results.MatchedUserFeedbackRules) != 0 {
			t.Fatal("inapplicable input triggered feedback inference, errors, or a match")
		}
	}
}

func TestFeedbackRejectsEmptyInputWithoutInventingSatisfaction(t *testing.T) {
	detector := &FeedbackDetector{initialized: true}
	for _, input := range []string{"", " \n\t"} {
		result, err := detector.Classify(context.Background(), input)
		if err == nil || result != nil {
			t.Fatalf("empty input returned a feedback classification: %+v, %v", result, err)
		}
	}
}

func TestShouldEvaluateUserFeedbackSignal(t *testing.T) {
	if shouldEvaluateUserFeedbackSignal(false) {
		t.Fatal("expected first-turn feedback evaluation to be skipped without a prior assistant reply")
	}
	if !shouldEvaluateUserFeedbackSignal(true) {
		t.Fatal("expected feedback evaluation to proceed when a prior assistant reply exists")
	}
}
