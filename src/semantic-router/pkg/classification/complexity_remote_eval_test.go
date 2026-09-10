package classification

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type stubScorer struct {
	score float64
	err   error
	calls int
}

func (s *stubScorer) Score(ctx context.Context, text string) (float64, error) {
	s.calls++
	if s.err != nil {
		return 0, s.err
	}
	return s.score, nil
}

func scoreRules() []config.ComplexityRule {
	hardAbove, easyBelow := 0.85, 0.60
	wideHard, wideEasy := 0.95, 0.30
	return []config.ComplexityRule{
		{Name: "needs_reasoning", HardAbove: &hardAbove, EasyBelow: &easyBelow},
		{Name: "extreme", HardAbove: &wideHard, EasyBelow: &wideEasy},
	}
}

// One remote call serves every rule: the score is a property of the request,
// and each rule only differs in where it draws its boundaries.
func TestComplexityRemoteScore_OneCallServesEveryRule(t *testing.T) {
	scorer := &stubScorer{score: 0.90}

	results, err := evaluateComplexityScore(context.Background(), scorer, "solve this", scoreRules())
	if err != nil {
		t.Fatalf("evaluateComplexityScore: %v", err)
	}

	if scorer.calls != 1 {
		t.Errorf("remote calls = %d, want 1 for two rules", scorer.calls)
	}
	if len(results) != 2 {
		t.Fatalf("results = %d, want one per rule", len(results))
	}
	// 0.90 is past needs_reasoning's hard boundary but inside extreme's
	// medium band, so the same score reaches two different verdicts.
	if results[0].Difficulty != config.ComplexityDifficultyHard {
		t.Errorf("needs_reasoning = %q, want hard", results[0].Difficulty)
	}
	if results[1].Difficulty != config.ComplexityDifficultyMedium {
		t.Errorf("extreme = %q, want medium", results[1].Difficulty)
	}
}

// score.v1 reports no confidence: a score just short of the hard boundary is
// the least certain position, not a strong one, so deriving a confidence from
// it would invent a number the model never reported.
func TestComplexityRemoteScore_ReportsNoConfidence(t *testing.T) {
	results, err := evaluateComplexityScore(context.Background(), &stubScorer{score: 0.90}, "solve this", scoreRules())
	if err != nil {
		t.Fatalf("evaluateComplexityScore: %v", err)
	}

	for _, result := range results {
		if result.ConfidenceReported {
			t.Errorf("rule %q reported a confidence on the score contract", result.RuleName)
		}
	}
}

// The raw score is published as it arrived, in the model's own units, rather
// than rescaled onto the local margin.
func TestComplexityRemoteScore_PublishesTheRawScore(t *testing.T) {
	results, err := evaluateComplexityScore(context.Background(), &stubScorer{score: 42.5}, "solve this", scoreRules())
	if err != nil {
		t.Fatalf("evaluateComplexityScore: %v", err)
	}

	if results[0].FusedMargin != 42.5 {
		t.Fatalf("published value = %v, want the raw 42.5", results[0].FusedMargin)
	}
}

func TestComplexityRemoteScore_PropagatesBackendFailure(t *testing.T) {
	wantErr := errors.New("scorer unreachable")

	if _, err := evaluateComplexityScore(context.Background(), &stubScorer{err: wantErr}, "solve this", scoreRules()); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want it to wrap the backend failure", err)
	}
}

// A malformed boundary pair is a config error, and it must surface rather than
// silently produce a verdict from zero values.
func TestComplexityRemoteScore_RejectsUnusableBoundaries(t *testing.T) {
	overlapHard, overlapEasy := 0.60, 0.85
	rules := []config.ComplexityRule{
		{Name: "broken", HardAbove: &overlapHard, EasyBelow: &overlapEasy},
	}

	if _, err := evaluateComplexityScore(context.Background(), &stubScorer{score: 0.7}, "solve this", rules); err == nil {
		t.Fatal("expected overlapping boundaries to be reported")
	}
}

type stubDistribution struct {
	result SequenceClassificationResult
	err    error
	calls  int
}

func (s *stubDistribution) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	s.calls++
	if s.err != nil {
		return SequenceClassificationResult{}, s.err
	}
	return s.result, nil
}

// On the label contract the winning label is the verdict, so no boundaries are
// consulted, and its probability is a real confidence. Probabilities arrive
// aligned to the declared label order, the same contract the generic
// classifier signal already uses for an inline mapping.
func TestComplexityRemoteLabels_WinningLabelIsTheVerdict(t *testing.T) {
	// Declared order: hard, easy, medium.
	backend := &stubDistribution{result: SequenceClassificationResult{
		Probabilities: []float32{0.92, 0.05, 0.03},
	}}

	results, err := evaluateComplexityLabels(context.Background(), backend, "solve this", scoreRules())
	if err != nil {
		t.Fatalf("evaluateComplexityLabels: %v", err)
	}

	if backend.calls != 1 {
		t.Errorf("remote calls = %d, want 1", backend.calls)
	}
	for _, result := range results {
		if result.Difficulty != config.ComplexityDifficultyHard {
			t.Errorf("rule %q = %q, want hard from the winning label", result.RuleName, result.Difficulty)
		}
		if !result.ConfidenceReported {
			t.Errorf("rule %q should report a confidence on the label contract", result.RuleName)
		}
		// Probabilities are float32 on the shared result type, so widening
		// cannot be exact.
		if diff := result.Confidence - 0.92; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("rule %q confidence = %v, want the winning label's probability", result.RuleName, result.Confidence)
		}
	}
}

// A distribution that does not match the verdict vocabulary cannot be turned
// into a verdict, so it fails rather than resolving to an empty one.
func TestComplexityRemoteLabels_RejectsAnUnusableDistribution(t *testing.T) {
	cases := map[string]SequenceClassificationResult{
		"empty":     {Probabilities: []float32{}},
		"too short": {Probabilities: []float32{0.9, 0.1}},
		"too long":  {Probabilities: []float32{0.4, 0.3, 0.2, 0.1}},
	}

	for name, result := range cases {
		backend := &stubDistribution{result: result}
		if _, err := evaluateComplexityLabels(context.Background(), backend, "solve this", scoreRules()); err == nil {
			t.Errorf("%s: expected an unusable distribution to be rejected", name)
		}
	}
}

func TestComplexityRemoteLabels_PropagatesBackendFailure(t *testing.T) {
	wantErr := errors.New("classifier unreachable")

	if _, err := evaluateComplexityLabels(context.Background(), &stubDistribution{err: wantErr}, "solve this", scoreRules()); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want it to wrap the backend failure", err)
	}
}
