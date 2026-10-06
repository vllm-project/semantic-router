package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var responseHallucinationTestRules = []config.HallucinationRule{{Name: "ungrounded_claims"}, {Name: "strict"}}

func TestEvaluateResponseHallucinationSignalPublishesNothingWithoutRules(t *testing.T) {
	if got := EvaluateResponseHallucinationSignal(nil, true, 1, ""); got != nil {
		t.Fatalf("got %+v", got)
	}
}

func TestEvaluateResponseHallucinationSignalDetectionMatchesEveryRule(t *testing.T) {
	signal := EvaluateResponseHallucinationSignal(responseHallucinationTestRules, true, 0.75, "")
	if len(signal.MatchedRules) != 2 || signal.MatchedRules[0] != "ungrounded_claims" || signal.MatchedRules[1] != "strict" {
		t.Fatalf("matched = %v", signal.MatchedRules)
	}
	if signal.Confidences["hallucination:strict"] != 0.75 || len(signal.Errors) != 0 {
		t.Fatalf("signal = %+v, want the confidence under every rule and no errors", signal)
	}
}

func TestEvaluateResponseHallucinationSignalCleanAnswerRecordsConfidence(t *testing.T) {
	signal := EvaluateResponseHallucinationSignal(responseHallucinationTestRules, false, 1, "")
	if len(signal.MatchedRules) != 0 || signal.Confidences["hallucination:ungrounded_claims"] != 1 {
		t.Fatalf("a clean answer must record its confidence and match nothing, got %+v", signal)
	}
}

func TestEvaluateResponseHallucinationSignalUnavailableCarriesTheCode(t *testing.T) {
	signal := EvaluateResponseHallucinationSignal(responseHallucinationTestRules, false, 0, HallucinationSignalContextUnavailable)
	if len(signal.MatchedRules) != 0 || len(signal.Confidences) != 0 {
		t.Fatalf("an unavailable answer must not read as clean, got %+v", signal)
	}
	for _, key := range []string{"hallucination:ungrounded_claims", "hallucination:strict"} {
		if signal.Errors[key] != HallucinationSignalContextUnavailable {
			t.Fatalf("errors = %v, want %s under %s", signal.Errors, HallucinationSignalContextUnavailable, key)
		}
	}
}
