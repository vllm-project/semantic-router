package tasks

import (
	"math"
	"testing"
)

func TestIndependentScoresPreserveMultiplePositiveLabels(t *testing.T) {
	for _, scores := range [][]float32{{.9, .8, .7}, {0, 0}, {1}} {
		if err := ValidateLabelScores(scores); err != nil {
			t.Fatal(err)
		}
	}
	for _, scores := range [][]float32{nil, {-1}, {1.01}, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if err := ValidateLabelScores(scores); err == nil {
			t.Fatalf("accepted invalid independent scores %v", scores)
		}
	}
}

func TestWindowCoverageRequiresCompleteOrderedTokenRanges(t *testing.T) {
	input := &InputUsage{OriginalTokens: 11, ProcessedTokens: 11}
	for _, spans := range [][][2]int{{{0, 5}, {3, 8}, {6, 9}}, {{0, 9}}} {
		if err := ValidateWindowCoverage(9, spans, input); err != nil {
			t.Fatal(err)
		}
	}
	for _, spans := range [][][2]int{nil, {{0, 5}, {0, 9}}, {{0, 5}, {3, 7}, {2, 9}}, {{1, 9}}, {{0, 5}, {6, 9}}, {{0, 5}, {3, 8}}, {{0, 5}, {2, 5}}, {{0, 10}}, {{0, 5}, {-1, 9}}} {
		if err := ValidateWindowCoverage(9, spans, input); err == nil {
			t.Fatalf("accepted incomplete coverage %v", spans)
		}
	}
	if err := ValidateWindowCoverage(0, [][2]int{{0, 0}}, &InputUsage{OriginalTokens: 2, ProcessedTokens: 2}); err != nil {
		t.Fatal(err)
	}
	if err := ValidateWindowCoverage(9, [][2]int{{0, 9}}, &InputUsage{OriginalTokens: 11, ProcessedTokens: 10, Truncated: true}); err == nil {
		t.Fatal("accepted truncated window metadata")
	}
}
