package tasks

import "testing"

func TestTokenSpansDoNotInventUnreportedProbabilities(t *testing.T) {
	available, unavailable := true, false
	for _, tc := range []struct {
		name         string
		availability *bool
		want         bool
	}{
		{"reported", &available, true},
		{"unavailable", &unavailable, false},
		{"undeclared", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result := TokenClassificationResult{
				Entities:        []TokenEntity{{EntityType: "unsupported", Text: "claim", Confidence: 0}},
				ScoresAvailable: tc.availability,
			}
			if result.HasScores() != tc.want {
				t.Fatalf("score availability became a fabricated probability: %+v", result)
			}
		})
	}
}
