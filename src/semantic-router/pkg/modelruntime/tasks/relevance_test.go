package tasks

import (
	"math"
	"testing"
)

func TestRelevanceScoresPreserveRawUnitsAndCompletePairs(t *testing.T) {
	input := []QueryDocument{{Query: "q", Document: "d"}, {Query: "q", Document: "other"}}
	valid := RelevanceScores{Scores: []float32{-2, 9}, Inputs: []InputUsage{{OriginalTokens: 7, ProcessedTokens: 7}, {OriginalTokens: 5, ProcessedTokens: 5}}}
	if err := ValidateRelevanceScores(input, valid); err != nil {
		t.Fatal(err)
	}
	if valid.InputMetadata().ProcessedTokens != 12 {
		t.Fatal("batch work was not preserved")
	}
	for _, bad := range []RelevanceScores{
		{Scores: []float32{1}, Inputs: valid.Inputs},
		{Scores: []float32{1, float32(math.NaN())}, Inputs: valid.Inputs},
		{Scores: valid.Scores, Inputs: []InputUsage{{OriginalTokens: 8, ProcessedTokens: 7}, {OriginalTokens: 5, ProcessedTokens: 5}}},
		{Scores: valid.Scores, Inputs: []InputUsage{{OriginalTokens: 7, ProcessedTokens: 7, Truncated: true}, {OriginalTokens: 5, ProcessedTokens: 5}}},
	} {
		if err := ValidateRelevanceScores(input, bad); err == nil {
			t.Fatalf("accepted %+v", bad)
		}
	}
}
