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
	if bestType != jailbreakClassificationErrorType {
		t.Errorf("bestType = %q, want %q", bestType, jailbreakClassificationErrorType)
	}
}

// TestFindBestJailbreakMatch_OnErrorBlock_DoesNotOverrideARealMatch verifies
// that when a genuine content-based match already scores above the
// fail-closed sentinel would, findBestJailbreakMatch still surfaces the
// contract callers rely on: a score of 1.0 either way, so a decision
// threshold at or below 1.0 behaves identically regardless of which content
// piece is examined first.
func TestFindBestJailbreakMatch_OnErrorBlock_DoesNotOverrideARealMatch(t *testing.T) {
	classifier := newJailbreakTestClassifier(t, config.OnErrorBlock)
	rule := config.JailbreakRule{Name: "default", Threshold: 0.5}
	cache := map[string][]cachedJailbreakResult{
		"attack text": {{result: SequenceClassificationResult{Probabilities: []float32{0.0, 1.0}}}},
		"broken text": {{err: errClassifyUnreachable}},
	}

	bestType, bestScore := classifier.findBestJailbreakMatch(
		rule, []string{"attack text", "broken text"}, cache,
	)

	if bestScore != 1.0 {
		t.Errorf("bestScore = %v, want 1.0", bestScore)
	}
	if bestType != jailbreakClassificationErrorType {
		t.Errorf("bestType = %q, want %q (fail-closed short-circuits the scan)", bestType, jailbreakClassificationErrorType)
	}
}

var errClassifyUnreachable = &jailbreakTestError{"classifier endpoint unreachable"}

type jailbreakTestError struct{ msg string }

func (e *jailbreakTestError) Error() string { return e.msg }
