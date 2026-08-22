package classification

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestFindBestJailbreakMatch_MultiplePositiveLabels guards against the
// signal-evaluation path (used by routing decisions) disagreeing with
// CheckForJailbreakWithRisk on multi-label configs: no single positive label
// wins argmax, but positive_labels' combined probability mass exceeds the
// rule's threshold. Thresholding argmax confidence instead of the summed
// positive-label mass would silently drop this match.
func TestFindBestJailbreakMatch_MultiplePositiveLabels(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.ModelID = "test-model"
	cfg.PromptGuard.JailbreakMappingPath = "test-mapping"
	cfg.PromptGuard.PositiveLabels = []string{"jailbreak", "INJECTION"}

	mapping := &JailbreakMapping{
		LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1, "INJECTION": 2},
		IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak", "2": "INJECTION"},
	}
	classifier, err := newClassifierWithOptions(cfg,
		withJailbreak(mapping, &MockJailbreakInitializer{}, &MockJailbreakInference{
			responseMap: make(map[string]MockJailbreakInferenceResponse),
		}))
	if err != nil {
		t.Fatalf("failed to construct classifier: %v", err)
	}

	rule := config.JailbreakRule{Name: "default", Threshold: 0.5}
	// Argmax is "benign" (0.40), but jailbreak (0.35) + INJECTION (0.25) = 0.60
	// combined risk, above the rule's threshold.
	cache := map[string][]cachedJailbreakResult{
		"some text": {
			{result: SequenceClassificationResult{
				Probabilities: []float32{0.40, 0.35, 0.25},
			}},
		},
	}

	bestType, bestScore := classifier.findBestJailbreakMatch(rule, []string{"some text"}, cache)

	if bestScore < rule.Threshold {
		t.Errorf("bestScore = %v, want >= %v (combined positive-label risk)", bestScore, rule.Threshold)
	}
	if math.Abs(float64(bestScore-0.60)) > 1e-6 {
		t.Errorf("bestScore = %v, want 0.60", bestScore)
	}
	if bestType != "benign" {
		t.Errorf("bestType = %q, want %q (argmax winner, for display)", bestType, "benign")
	}
}

// newJailbreakTestClassifier builds a minimal *Classifier for
// findBestJailbreakMatch tests, with the given PromptGuard.OnError policy.
func newJailbreakTestClassifier(t *testing.T, onError string) *Classifier {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.ModelID = "test-model"
	cfg.PromptGuard.JailbreakMappingPath = "test-mapping"
	cfg.PromptGuard.PositiveLabels = []string{"jailbreak"}
	cfg.PromptGuard.OnError = onError

	mapping := &JailbreakMapping{
		LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1},
		IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"},
	}
	classifier, err := newClassifierWithOptions(cfg,
		withJailbreak(mapping, &MockJailbreakInitializer{}, &MockJailbreakInference{
			responseMap: make(map[string]MockJailbreakInferenceResponse),
		}))
	if err != nil {
		t.Fatalf("failed to construct classifier: %v", err)
	}
	return classifier
}

// TestFindBestJailbreakMatch_OnErrorAllow_DefaultToleratesFailure guards the
// historical, default behavior: a classify error is logged and the affected
// content is treated as not matching, so a request whose only content piece
// failed to classify does not report a match.
func TestFindBestJailbreakMatch_OnErrorAllow_DefaultToleratesFailure(t *testing.T) {
	classifier := newJailbreakTestClassifier(t, "")
	rule := config.JailbreakRule{Name: "default", Threshold: 0.5}
	cache := map[string][]cachedJailbreakResult{
		"some text": {{err: errClassifyUnreachable}},
	}

	bestType, bestScore := classifier.findBestJailbreakMatch(rule, []string{"some text"}, cache)

	if bestScore != 0 {
		t.Errorf("bestScore = %v, want 0 (on_error: allow tolerates the failure)", bestScore)
	}
	if bestType != "" {
		t.Errorf("bestType = %q, want empty", bestType)
	}
}

// TestFindBestJailbreakMatch_OnErrorBlock_TreatsFailureAsMatch verifies
// on_error: block closes the gap @adaamko flagged on #2760: an
// unreachable classifier endpoint must not look identical to a genuinely
// clean request. With on_error: block, the rule reports a match at maximum
// confidence instead of silently skipping the failed content.
func TestFindBestJailbreakMatch_OnErrorBlock_TreatsFailureAsMatch(t *testing.T) {
	classifier := newJailbreakTestClassifier(t, config.OnErrorBlock)
	rule := config.JailbreakRule{Name: "default", Threshold: 0.5}
	cache := map[string][]cachedJailbreakResult{
		"some text": {{err: errClassifyUnreachable}},
	}

	bestType, bestScore := classifier.findBestJailbreakMatch(rule, []string{"some text"}, cache)

	if bestScore != 1.0 {
		t.Errorf("bestScore = %v, want 1.0 (fail-closed)", bestScore)
	}
	if bestType != JailbreakClassificationErrorType {
		t.Errorf("bestType = %q, want %q", bestType, JailbreakClassificationErrorType)
	}
}

// TestFindBestJailbreakMatch_OnErrorBlock_DoesNotOverrideARealMatch verifies
// that a genuine content-based detection is reported with its own label and
// score even when another content piece failed to classify under
// on_error: block. Both outcomes block the request, but reporting the
// sentinel here would erase the real label and score from
// results.JailbreakType/JailbreakConfidence, the replay record's
// jailbreak_type, and the jailbreak.type span attribute - making a replayed
// real attack indistinguishable from a guardrail outage.
//
// The real score is deliberately 0.97, not 1.0: with a 1.0 fixture the
// sentinel's own 1.0 makes the substitution invisible, which is exactly how
// an earlier version of this test asserted the opposite of its own name
// without anyone noticing.
func TestFindBestJailbreakMatch_OnErrorBlock_DoesNotOverrideARealMatch(t *testing.T) {
	classifier := newJailbreakTestClassifier(t, config.OnErrorBlock)
	rule := config.JailbreakRule{Name: "default", Threshold: 0.5}
	cache := map[string][]cachedJailbreakResult{
		"attack text": {{result: SequenceClassificationResult{Probabilities: []float32{0.03, 0.97}}}},
		"broken text": {{err: errClassifyUnreachable}},
	}

	// Both orderings, because the fail-closed error must not win by being
	// encountered first.
	for _, order := range [][]string{
		{"attack text", "broken text"},
		{"broken text", "attack text"},
	} {
		bestType, bestScore := classifier.findBestJailbreakMatch(rule, order, cache)

		if bestType != "jailbreak" {
			t.Errorf("order %v: bestType = %q, want %q (a real detection must not be replaced by the sentinel)",
				order, bestType, "jailbreak")
		}
		if bestScore != 0.97 {
			t.Errorf("order %v: bestScore = %v, want 0.97 (the real score must survive)", order, bestScore)
		}
	}
}

// TestFindBestJailbreakMatch_OnErrorBlock_FailsClosedOnUninterpretableResult
// covers the other way a result can be unusable: the call succeeded, but its
// argmax index has no entry in the configured mapping (reachable because
// nothing validates the model head against the mapping - the initializers
// ignore numClasses and the FFI reports the model's own head size). That is
// "could not verify safe" just like a transport error, so on_error: block
// must close rather than treat it as clean.
func TestFindBestJailbreakMatch_OnErrorBlock_FailsClosedOnUninterpretableResult(t *testing.T) {
	classifier := newJailbreakTestClassifier(t, config.OnErrorBlock)
	rule := config.JailbreakRule{Name: "default", Threshold: 0.5}
	// Mapping under test has 2 labels; argmax here is index 2.
	cache := map[string][]cachedJailbreakResult{
		"odd text": {{result: SequenceClassificationResult{Probabilities: []float32{0.10, 0.20, 0.70}}}},
	}

	bestType, bestScore := classifier.findBestJailbreakMatch(rule, []string{"odd text"}, cache)

	if bestType != JailbreakClassificationErrorType {
		t.Errorf("bestType = %q, want %q (an uninterpretable result must fail closed under block)",
			bestType, JailbreakClassificationErrorType)
	}
	if bestScore != 1.0 {
		t.Errorf("bestScore = %v, want 1.0", bestScore)
	}
}

// TestFindBestJailbreakMatch_OnErrorAllow_ToleratesUninterpretableResult is
// the default-path counterpart: on_error: allow must keep its historical
// behaviour of logging and moving on.
func TestFindBestJailbreakMatch_OnErrorAllow_ToleratesUninterpretableResult(t *testing.T) {
	classifier := newJailbreakTestClassifier(t, config.OnErrorAllow)
	rule := config.JailbreakRule{Name: "default", Threshold: 0.5}
	cache := map[string][]cachedJailbreakResult{
		"odd text": {{result: SequenceClassificationResult{Probabilities: []float32{0.10, 0.20, 0.70}}}},
	}

	bestType, bestScore := classifier.findBestJailbreakMatch(rule, []string{"odd text"}, cache)

	if bestType != "" || bestScore != 0 {
		t.Errorf("bestType, bestScore = %q, %v; want \"\", 0", bestType, bestScore)
	}
}

// TestFindBestJailbreakMatch_OnErrorBlock_FailsClosedWhenNoRealMatch is the
// other half of the contract above: with no genuine detection to report, a
// classify failure under on_error: block must still block, via the sentinel.
func TestFindBestJailbreakMatch_OnErrorBlock_FailsClosedWhenNoRealMatch(t *testing.T) {
	classifier := newJailbreakTestClassifier(t, config.OnErrorBlock)
	rule := config.JailbreakRule{Name: "default", Threshold: 0.5}
	cache := map[string][]cachedJailbreakResult{
		"benign text": {{result: SequenceClassificationResult{Probabilities: []float32{0.98, 0.02}}}},
		"broken text": {{err: errClassifyUnreachable}},
	}

	bestType, bestScore := classifier.findBestJailbreakMatch(
		rule, []string{"benign text", "broken text"}, cache,
	)

	if bestType != JailbreakClassificationErrorType {
		t.Errorf("bestType = %q, want %q", bestType, JailbreakClassificationErrorType)
	}
	if bestScore != 1.0 {
		t.Errorf("bestScore = %v, want 1.0", bestScore)
	}
}

var errClassifyUnreachable = &jailbreakTestError{"classifier endpoint unreachable"}

type jailbreakTestError struct{ msg string }

func (e *jailbreakTestError) Error() string { return e.msg }
