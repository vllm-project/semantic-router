package config

import (
	"reflect"
	"testing"
)

func leafDecision(name string, tier int, signalType string) Decision {
	return Decision{Name: name, Tier: tier, Rules: RuleNode{Type: signalType, Name: "rule"}}
}

func TestAmbiguousConfidencePools(t *testing.T) {
	for _, testCase := range []struct {
		name      string
		decisions []Decision
		want      []confidencePoolFallback
	}{
		{
			name: "one kind per tier",
			decisions: []Decision{
				leafDecision("law", 1, SignalTypeDomain),
				leafDecision("health", 1, SignalTypeDomain),
			},
		},
		{
			name: "probability against similarity",
			decisions: []Decision{
				leafDecision("semantic", 2, SignalTypeEmbedding),
				leafDecision("law", 2, SignalTypeDomain),
			},
			want: []confidencePoolFallback{{
				Pool:      2,
				Decisions: []string{"law", "semantic"},
				Kinds:     []string{string(ScoreKindProbability), string(ScoreKindSimilarity)},
			}},
		},
		{
			name: "untiered decisions share one pool",
			decisions: []Decision{
				leafDecision("semantic", 0, SignalTypeEmbedding),
				leafDecision("law", 0, SignalTypeDomain),
				{Name: "fallback", Rules: RuleNode{Operator: RuleOperatorAnd}},
			},
			want: []confidencePoolFallback{{
				Pool:      0,
				Decisions: []string{"law", "semantic"},
				Kinds:     []string{string(ScoreKindProbability), string(ScoreKindSimilarity)},
			}},
		},
		{
			name: "policy leaves declare no kind",
			decisions: []Decision{
				leafDecision("omni", 1, SignalTypeConversation),
				leafDecision("projected", 1, SignalTypeProjection),
			},
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			got := ambiguousConfidencePools(testCase.decisions)
			if len(got) == 0 && len(testCase.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, testCase.want) {
				t.Fatalf("ambiguousConfidencePools() = %+v, want %+v", got, testCase.want)
			}
		})
	}
}

func TestSignalScoreKind(t *testing.T) {
	for signalType, want := range map[string]ScoreKind{
		SignalTypeDomain:       ScoreKindProbability,
		SignalTypeEmbedding:    ScoreKindSimilarity,
		SignalTypeReask:        ScoreKindSimilarity,
		SignalTypeKeyword:      ScoreKindNone,
		SignalTypeConversation: ScoreKindNone,
		SignalTypeProjection:   ScoreKindNone,
		SignalTypeMetadata:     ScoreKindNone,
		"unregistered":         ScoreKindNone,
	} {
		if got := SignalScoreKind(signalType); got != want {
			t.Errorf("SignalScoreKind(%q) = %q, want %q", signalType, got, want)
		}
	}
}
