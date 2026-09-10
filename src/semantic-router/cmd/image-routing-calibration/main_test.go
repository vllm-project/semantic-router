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

// Two bands with different confusion matrices but the same mathematical F1
// (1/3 each). Through precision and recall the floats differ in the last bit,
// so a float comparison never reaches the width tie-break and picks the
// narrow (0, 0.1] band; the rational comparison must pick (0.15, 0.7].
func TestSelectThreshold_TieUsesExactF1(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{
		"p1": 0.1, "p2": 0.7,
		"n1": 0.11, "n2": 0.12, "n3": 0.13, "n4": 0.14, "n5": 0.15, "n6": 0.8, "n7": 0.85, "n8": 0.9,
	}, "p1", "p2")
	report := calibrate(t, fixtures, positive)

	// The wider band is (0.15, 0.7]; its midpoint 0.425 rounds to 0.42, which
	// yields the same matrix and is therefore what the selector reports.
	if got := report.Selected.Threshold; got <= 0.15 || got > 0.7 {
		t.Fatalf("selected threshold = %.4f, want a value inside the wider equal-F1 band (0.15, 0.7]", got)
	}
	if report.Selected.TP != 1 || report.Selected.FP != 3 || report.Selected.FN != 1 {
		t.Fatalf("selected matrix = TP %d FP %d FN %d, want 1/3/1", report.Selected.TP, report.Selected.FP, report.Selected.FN)
	}
}

// A positive scoring exactly at the floor is only admitted by the boundary
// threshold itself; the band midpoints all miss it.
func TestSelectThreshold_ZeroScoredPositiveUsesBoundary(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{"p1": 0, "n1": 0.5}, "p1")
	report := calibrate(t, fixtures, positive)

	if report.Selected.Threshold != 0 || report.Selected.TP != 1 || report.Selected.FP != 1 {
		t.Fatalf("selected = %+v, want threshold 0 with TP 1 FP 1", report.Selected)
	}
}

// One distinct score means a one-value sweep; the reported matrix must be the
// evaluation of that threshold, not an empty result.
func TestSelectThreshold_SingletonSweepIsEvaluated(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{"p1": 0, "n1": 0}, "p1")
	report := calibrate(t, fixtures, positive)

	if len(report.Sweep) != 1 {
		t.Fatalf("sweep has %d entries, want 1", len(report.Sweep))
	}
	if report.Selected.Threshold != 0 || report.Selected.TP != 1 || report.Selected.FP != 1 {
		t.Fatalf("selected = %+v, want threshold 0 with TP 1 FP 1", report.Selected)
	}
}

// A negative score sits below the boundary; the boundary candidate is the
// lowest sweep value, so it admits the fixture instead of reporting F1 0
// from a band midpoint that misses it.
func TestSelectThreshold_NegativeScoreBoundary(t *testing.T) {
	fixtures, positive := scoredFixtures(map[string]float64{"p1": -0.2, "n1": 0.5}, "p1")
	report := calibrate(t, fixtures, positive)

	if report.Selected.Threshold != -0.2 || report.Selected.TP != 1 || report.Selected.FP != 1 {
		t.Fatalf("selected = %+v, want the boundary threshold -0.2 with TP 1 FP 1", report.Selected)
	}
}

func parseTestSet(t *testing.T, manifest string) error {
	t.Helper()
	_, err := parseSet([]byte(manifest), []config.EmbeddingRule{{Name: testRule}})
	return err
}

// The manifest is explicit: every fixture carries a reviewed label, a path
// cannot carry two labels, and an exclusion must say why.
func TestParseSet_AcceptsReviewedManifest(t *testing.T) {
	err := parseTestSet(t, `{
		"positives": [{"image_file": "a.png", "signal_name": "rule"}],
		"negatives": ["b.jpg", "c.jpeg"],
		"excluded": [{"image_file": "d.png", "reason": "ambiguous"}]
	}`)
	if err != nil {
		t.Fatalf("valid manifest rejected: %v", err)
	}
}

func TestParseSet_RejectsUnreviewedShapes(t *testing.T) {
	cases := map[string]string{
		"no negatives":        `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": []}`,
		"no positives":        `{"positives": [], "negatives": ["b.png"]}`,
		"unknown rule":        `{"positives": [{"image_file": "a.png", "signal_name": "other"}], "negatives": ["b.png"]}`,
		"positive twice":      `{"positives": [{"image_file": "a.png", "signal_name": "rule"}, {"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png"]}`,
		"positive+negative":   `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["a.png"]}`,
		"negative twice":      `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png", "b.png"]}`,
		"negative+excluded":   `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png"], "excluded": [{"image_file": "b.png", "reason": "x"}]}`,
		"exclusion no reason": `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.png"], "excluded": [{"image_file": "c.png"}]}`,
		"bad extension":       `{"positives": [{"image_file": "a.png", "signal_name": "rule"}], "negatives": ["b.gif"]}`,
		"empty path":          `{"positives": [{"image_file": "", "signal_name": "rule"}], "negatives": ["b.png"]}`,
		"legacy roots":        `{"negative_roots": ["images"], "positives": [{"image_file": "a.png", "signal_name": "rule"}]}`,
	}
	for name, manifest := range cases {
		t.Run(name, func(t *testing.T) {
			if err := parseTestSet(t, manifest); err == nil {
				t.Fatalf("manifest accepted: %s", manifest)
			}
		})
	}
}
