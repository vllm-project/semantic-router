package classification

import "testing"

func TestShouldEvaluateUserFeedbackSignal(t *testing.T) {
	if shouldEvaluateUserFeedbackSignal(false) {
		t.Fatal("expected first-turn feedback evaluation to be skipped without a prior assistant reply")
	}
	if !shouldEvaluateUserFeedbackSignal(true) {
		t.Fatal("expected feedback evaluation to proceed when a prior assistant reply exists")
	}
}
