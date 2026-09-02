package classification

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEvaluateResponseJailbreakSignal(t *testing.T) {
	rules := []config.ResponseJailbreakRule{
		{Name: "strict", Threshold: 0.5},
		{Name: "lenient", Threshold: 0.9},
	}

	signal := EvaluateResponseJailbreakSignal(rules, 0.7, true)
	if signal == nil {
		t.Fatal("declared rules must produce a signal")
	}
	if len(signal.MatchedRules) != 1 || signal.MatchedRules[0] != "strict" {
		t.Errorf("matched = %v, want only strict at risk 0.7", signal.MatchedRules)
	}
	// Every rule reports the score it thresholded, matched or not, so a
	// predicate leaf can read the value without the rule having matched.
	if got := signal.Confidences["response_jailbreak:lenient"]; math.Abs(got-0.7) > 1e-6 {
		t.Errorf("lenient confidence = %v, want 0.7 even though it did not match", got)
	}
	if len(signal.Errors) != 0 {
		t.Errorf("a resolved detector must not report errors: %v", signal.Errors)
	}
}

// An unresolved detector is not a clean response. It has to reach the decision
// as an error, so on_unknown decides what an unverified response means, rather
// than every rule quietly reporting no match.
func TestEvaluateResponseJailbreakSignalUnresolved(t *testing.T) {
	rules := []config.ResponseJailbreakRule{{Name: "strict", Threshold: 0.5}}

	signal := EvaluateResponseJailbreakSignal(rules, 0, false)
	if signal == nil {
		t.Fatal("declared rules must produce a signal")
	}
	if len(signal.MatchedRules) != 0 {
		t.Errorf("an unresolved detector must not match: %v", signal.MatchedRules)
	}
	if len(signal.Confidences) != 0 {
		t.Errorf("an unresolved detector has no score to report: %v", signal.Confidences)
	}
	if signal.Errors["response_jailbreak:strict"] != responseJailbreakSignalFailedCode {
		t.Errorf("errors = %v, want the rule recorded as unresolved", signal.Errors)
	}
}

func TestEvaluateResponseJailbreakSignalWithoutRules(t *testing.T) {
	if signal := EvaluateResponseJailbreakSignal(nil, 0.9, true); signal != nil {
		t.Errorf("no declared rules must produce no signal, got %+v", signal)
	}
}
