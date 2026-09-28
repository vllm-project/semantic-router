package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The local path has always reported |margin| as its confidence, and the
// decision engine, numeric predicates, projections and router-learning all
// read it. Introducing ConfidenceReported for score.v1 must not take that
// away from every config that has no backend at all.
func TestLocalComplexityResultReportsItsConfidence(t *testing.T) {
	classifier := &ComplexityClassifier{
		rules: []config.ComplexityRule{{Name: "needs_reasoning", Threshold: 0.10}},
		hardPrototypeBanks: map[string]*prototypeBank{
			"needs_reasoning": newPrototypeBank(nil, config.PrototypeScoringConfig{}.WithDefaults()),
		},
		easyPrototypeBanks: map[string]*prototypeBank{
			"needs_reasoning": newPrototypeBank(nil, config.PrototypeScoringConfig{}.WithDefaults()),
		},
		prototypeCfg: config.PrototypeScoringConfig{}.WithDefaults(),
	}

	result := classifier.classifyRuleWithEmbeddings(
		classifier.rules[0],
		complexityQueryEmbeddings{text: []float32{1, 0, 0}},
		defaultPrototypeScoreOptions(classifier.prototypeCfg),
	)

	if !result.ConfidenceReported {
		t.Fatal("the local prototype path must report its confidence; " +
			"leaving it unset drops complexity out of confidence-based decision ranking, " +
			"numeric predicates, projections and router-learning outcomes")
	}
}

// The local path must honour an explicit boundary pair, not silently fall back
// to threshold: 0. The docs and the shipped reference config both state that
// the asymmetric form works locally, and a rule declaring cut points that are
// discarded makes medium unreachable while looking correct.
func TestLocalComplexityHonoursAnExplicitBoundaryPair(t *testing.T) {
	hardAbove, easyBelow := 0.60, -0.20
	rule := config.ComplexityRule{
		Name:      "asymmetric",
		HardAbove: &hardAbove,
		EasyBelow: &easyBelow,
	}

	bounds, err := rule.EffectiveBoundaries()
	if err != nil {
		t.Fatalf("EffectiveBoundaries: %v", err)
	}

	// A margin inside the declared band must be medium. With the pair
	// discarded and threshold defaulting to 0, this would be hard.
	if got := localComplexityVerdict(bounds, 0.45); got != config.ComplexityDifficultyMedium {
		t.Errorf("margin 0.45 = %q, want medium: it sits below hard_above (0.60)", got)
	}
	if got := localComplexityVerdict(bounds, 0.75); got != config.ComplexityDifficultyHard {
		t.Errorf("margin 0.75 = %q, want hard", got)
	}
	if got := localComplexityVerdict(bounds, -0.30); got != config.ComplexityDifficultyEasy {
		t.Errorf("margin -0.30 = %q, want easy", got)
	}
}

// The shorthand must keep behaving exactly as it always has.
func TestLocalComplexityThresholdShorthandIsUnchanged(t *testing.T) {
	bounds, err := config.ComplexityRule{Name: "classic", Threshold: 0.10}.EffectiveBoundaries()
	if err != nil {
		t.Fatalf("EffectiveBoundaries: %v", err)
	}

	cases := map[float64]string{
		0.50:  config.ComplexityDifficultyHard,
		0.05:  config.ComplexityDifficultyMedium,
		-0.05: config.ComplexityDifficultyMedium,
		-0.50: config.ComplexityDifficultyEasy,
	}
	for margin, want := range cases {
		if got := localComplexityVerdict(bounds, margin); got != want {
			t.Errorf("margin %v = %q, want %q", margin, got, want)
		}
	}
}
