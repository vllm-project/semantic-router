package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// newMultiEmitTestClassifier builds a classifier with one weighted_sum score and
// one mapping using the given method plus three deliberately overlapping output
// bands. For a score of 0.6: low_band (lt 0.3) does NOT match, while both
// mid_band ([0.2,0.7)) and high_band ([0.5,inf)) match, letting first-hit and
// multi_emit behavior be distinguished.
func newMultiEmitTestClassifier(method string) *Classifier {
	return &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "multi_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{{
							Type:        config.SignalTypeKeyword,
							Name:        "test_signal",
							Weight:      1.0,
							ValueSource: "confidence",
						}},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "multi_mapping",
						Source: "multi_score",
						Method: method,
						Outputs: []config.ProjectionMappingOutput{
							{Name: "low_band", LT: float64Ptr(0.3)},
							{Name: "mid_band", GTE: float64Ptr(0.2), LT: float64Ptr(0.7)},
							{Name: "high_band", GTE: float64Ptr(0.5)},
						},
					}},
				},
			},
		},
	}
}

func multiEmitTestResults() *SignalResults {
	return &SignalResults{
		MatchedKeywordRules: []string{"test_signal"},
		SignalConfidences:   map[string]float64{"keyword:test_signal": 0.6},
	}
}

func assertMatchedProjectionRules(t *testing.T, got []string, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("matched projection rules = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("matched projection rules = %v, want %v", got, want)
		}
	}
}

func TestApplyProjectionsMultiEmitEmitsAllMatchingBands(t *testing.T) {
	classifier := newMultiEmitTestClassifier("multi_emit")

	got := classifier.applyProjections(multiEmitTestResults())
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}

	// score = 1.0*0.6 = 0.6 -> mid_band and high_band match; low_band does not.
	assertMatchedProjectionRules(t, got.MatchedProjectionRules, []string{"mid_band", "high_band"})
	for _, name := range []string{"mid_band", "high_band"} {
		if _, ok := got.SignalConfidences["projection:"+name]; !ok {
			t.Fatalf("missing projection confidence for %q in %+v", name, got.SignalConfidences)
		}
	}
	if got.ProjectionTrace == nil || len(got.ProjectionTrace.Mappings) != 1 {
		t.Fatalf("projection trace = %+v, want one mapping", got.ProjectionTrace)
	}
	mappingTrace := got.ProjectionTrace.Mappings[0]
	assertMatchedProjectionRules(t, mappingTrace.MatchedOutputs, []string{"mid_band", "high_band"})
	if mappingTrace.SelectedOutput != "mid_band" {
		t.Fatalf("selected output = %q, want first matched output mid_band", mappingTrace.SelectedOutput)
	}
}

func TestApplyProjectionsThresholdBandsEmitsFirstMatchOnly(t *testing.T) {
	classifier := newMultiEmitTestClassifier("threshold_bands")

	got := classifier.applyProjections(multiEmitTestResults())
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}

	// First-hit: only the first matching band (mid_band, in output order) is emitted.
	assertMatchedProjectionRules(t, got.MatchedProjectionRules, []string{"mid_band"})
}

func TestApplyProjectionsEmptyMethodDefaultsToFirstHit(t *testing.T) {
	classifier := newMultiEmitTestClassifier("")

	got := classifier.applyProjections(multiEmitTestResults())
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}

	// Unset method must preserve the legacy first-hit (threshold_bands) behavior.
	assertMatchedProjectionRules(t, got.MatchedProjectionRules, []string{"mid_band"})
}
