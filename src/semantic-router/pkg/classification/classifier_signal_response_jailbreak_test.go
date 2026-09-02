package classification

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEvaluateResponseJailbreakSignal(t *testing.T) {
	rules := []config.JailbreakRule{
		{Name: "strict", Threshold: 0.5, Direction: config.SignalDirectionResponse},
		{Name: "lenient", Threshold: 0.9, Direction: config.SignalDirectionResponse},
	}

	signal := EvaluateResponseJailbreakSignal(rules, 0.7, true)
	if signal == nil {
		t.Fatal("declared rules must produce a signal")
	}
	if len(signal.MatchedRules) != 1 || signal.MatchedRules[0] != "strict" {
		t.Errorf("matched = %v, want only strict at risk 0.7", signal.MatchedRules)
	}
	// Every rule reports the score it thresholded, matched or not, so a
	// predicate leaf can read the value without the rule having matched. The
	// key is the plain jailbreak key: the direction lives on the rule, not in
	// the name a decision reads it by.
	if got := signal.Confidences["jailbreak:lenient"]; math.Abs(got-0.7) > 1e-6 {
		t.Errorf("lenient confidence = %v, want 0.7 even though it did not match", got)
	}
	if len(signal.Errors) != 0 {
		t.Errorf("a resolved detector must not report errors: %v", signal.Errors)
	}
}

func TestEvaluateResponseJailbreakSignalUnresolved(t *testing.T) {
	rules := []config.JailbreakRule{{Name: "strict", Threshold: 0.5, Direction: config.SignalDirectionResponse}}

	signal := EvaluateResponseJailbreakSignal(rules, 0, false)
	if signal == nil {
		t.Fatal("an unresolved detector still produces a signal")
	}
	if len(signal.MatchedRules) != 0 {
		t.Errorf("an unresolved detector must not match: %v", signal.MatchedRules)
	}
	if _, ok := signal.Confidences["jailbreak:strict"]; ok {
		t.Error("an unresolved detector must not report a score that could be read as clean")
	}
	if got := signal.Errors["jailbreak:strict"]; got != responseJailbreakSignalFailedCode {
		t.Errorf("error = %q, want %q", got, responseJailbreakSignalFailedCode)
	}
}

func TestEvaluateResponseJailbreakSignalNoRules(t *testing.T) {
	if signal := EvaluateResponseJailbreakSignal(nil, 0.9, true); signal != nil {
		t.Errorf("no rules must produce no signal, got %+v", signal)
	}
}
