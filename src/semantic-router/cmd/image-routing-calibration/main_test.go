package main

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const testRule = "rule"

// scoredFixtures builds one fixture per score; the positive set is by path.
func scoredFixtures(scores map[string]float64, positives ...string) ([]fixtureReport, map[string]bool) {
	fixtures := make([]fixtureReport, 0, len(scores))
	for path, score := range scores {
		fixtures = append(fixtures, fixtureReport{Path: path, Scores: map[string]float64{testRule: score}})
	}
	positive := map[string]bool{}
	for _, path := range positives {
		positive[path] = true
	}
	return fixtures, positive
}

func calibrate(t *testing.T, fixtures []fixtureReport, positive map[string]bool) ruleReport {
	t.Helper()
	for i := range fixtures {
		if positive[fixtures[i].Path] {
			fixtures[i].PositiveFor = []string{testRule}
		}
	}
	return calibrateRule(config.EmbeddingRule{Name: testRule, SimilarityThreshold: 0.5}, fixtures)
}

// Two max-F1 bands with equal F1 but different widths: (0.30, 0.50] is 0.20
// wide, (0.85, 0.90] is 0.05 wide. The global margin would prefer the narrow
// band (it sits closer to max(negative)); the documented rule prefers the
// wider one, so the midpoint 0.40 must win.
func TestSelectThreshold_TieBreaksTowardWiderBand(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"neg-a": 0.10, "neg-b": 0.30, "pos-a": 0.50, "neg-c": 0.70, "neg-d": 0.85, "pos-b": 0.90,
	}, "pos-a", "pos-b")
	report := calibrate(t, fixtures, positive)

	if report.Selected.Separable {
		t.Fatalf("fixture set is interleaved; expected non-separable, got %+v", report.Selected)
	}
	if got, want := report.Selected.Threshold, 0.40; math.Abs(got-want) > 1e-9 {
		t.Fatalf("selected threshold = %.4f, want %.4f (midpoint of the wider max-F1 band)", got, want)
	}
	if got, want := report.Selected.F1, 2.0/3.0; math.Abs(got-want) > 1e-9 {
		t.Fatalf("selected F1 = %.4f, want %.4f", got, want)
	}
}

// Separable sets take the midpoint of the gap, rounded to two decimals when
// the rounded value still separates.
func TestSelectThreshold_SeparableMidpointRounds(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"neg-a": 0.20, "neg-b": 0.41, "pos-a": 0.63, "pos-b": 0.80,
	}, "pos-a", "pos-b")
	report := calibrate(t, fixtures, positive)

	if !report.Selected.Separable {
		t.Fatalf("expected separable selection, got %+v", report.Selected)
	}
	if got, want := report.Selected.Threshold, 0.52; math.Abs(got-want) > 1e-9 {
		t.Fatalf("selected threshold = %.4f, want %.4f (rounded gap midpoint)", got, want)
	}
}

// The sweep must not depend on the currently shipped threshold.
func TestCalibrateRule_SweepIgnoresShippedThreshold(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"neg-a": 0.10, "neg-b": 0.30, "pos-a": 0.50, "neg-c": 0.70, "neg-d": 0.85, "pos-b": 0.90,
	}, "pos-a", "pos-b")
	for i := range fixtures {
		if positive[fixtures[i].Path] {
			fixtures[i].PositiveFor = []string{testRule}
		}
	}
	a := calibrateRule(config.EmbeddingRule{Name: testRule, SimilarityThreshold: 0.44}, fixtures)
	b := calibrateRule(config.EmbeddingRule{Name: testRule, SimilarityThreshold: 0.88}, fixtures)
	if a.Selected.Threshold != b.Selected.Threshold || len(a.Sweep) != len(b.Sweep) {
		t.Fatalf("selection changed with the shipped value: %.4f (sweep %d) vs %.4f (sweep %d)",
			a.Selected.Threshold, len(a.Sweep), b.Selected.Threshold, len(b.Sweep))
	}
}
