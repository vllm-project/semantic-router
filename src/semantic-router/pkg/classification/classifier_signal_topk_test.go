package classification

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmbeddingOutputPreservesIndependentPredicatesForProjectionAndPriority(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"query": makeEmbedding(1, 0, 0),
		"one":   makeEmbedding(0.95, 0, 0),
		"two":   makeEmbedding(0.85, 0, 0),
		"three": makeEmbedding(0.75, 0, 0),
		"four":  makeEmbedding(0.4, 0, 0),
	})
	for _, tc := range []struct {
		name        string
		topK        *int
		wantMatches []string
		wantRoute   string
	}{
		{name: "default", wantMatches: []string{"first", "second", "third"}, wantRoute: "priority"},
		{name: "explicit_unlimited", topK: intPtr(0), wantMatches: []string{"first", "second", "third"}, wantRoute: "priority"},
		{name: "negative_fallback", topK: intPtr(-1), wantMatches: []string{"first", "second", "third"}, wantRoute: "priority"},
		{name: "explicit_one", topK: intPtr(1), wantMatches: []string{"first"}, wantRoute: "strongest"},
		{name: "explicit_two", topK: intPtr(2), wantMatches: []string{"first", "second"}, wantRoute: "priority"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rules := []config.EmbeddingRule{
				{Name: "first", Candidates: []string{"one"}, SimilarityThreshold: 0.7},
				{Name: "second", Candidates: []string{"two"}, SimilarityThreshold: 0.7},
				{Name: "third", Candidates: []string{"three"}, SimilarityThreshold: 0.7},
				{Name: "below", Candidates: []string{"four"}, SimilarityThreshold: 0.7},
			}
			options := config.HNSWConfig{TopK: tc.topK, PreloadEmbeddings: true}
			c := &Classifier{
				keywordEmbeddingClassifier: newTestEmbeddingClassifier(t, rules, options),
				Config: &config.RouterConfig{
					InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{EmbeddingConfig: options}},
					IntelligentRouting: config.IntelligentRouting{
						Signals: config.Signals{EmbeddingRules: rules},
						Projections: config.Projections{
							Scores: []config.ProjectionScore{{Name: "evidence", Method: "weighted_sum", Inputs: []config.ProjectionScoreInput{
								{Type: "embedding", Name: "second", ValueSource: "confidence", Weight: 1},
							}}},
							Mappings: []config.ProjectionMapping{{Name: "band", Source: "evidence", Method: "threshold_bands", Outputs: []config.ProjectionMappingOutput{
								{Name: "qualified", GTE: float64Ptr(0.7)},
							}}},
						},
						Strategy: config.RoutingStrategyPriority,
						Decisions: []config.Decision{
							{Name: "negative", Priority: 30, Rules: config.RuleNode{Type: "embedding", Name: "below"}},
							{Name: "priority", Priority: 20, Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{
								{Type: "embedding", Name: "second"}, {Type: "projection", Name: "qualified"},
							}}},
							{Name: "strongest", Priority: 10, Rules: config.RuleNode{Type: "embedding", Name: "first"}},
							{Name: "third", Priority: 1, Rules: config.RuleNode{Type: "embedding", Name: "third"}},
						},
					},
				},
			}
			matches, err := c.keywordEmbeddingClassifier.ClassifyAll("query")
			require.NoError(t, err)
			names := make([]string, 0, len(matches))
			for _, match := range matches {
				names = append(names, match.RuleName)
			}
			require.ElementsMatch(t, tc.wantMatches, names, "direct classification and signal emission must use the same limit")
			signals := c.EvaluateAllSignals("query")
			require.ElementsMatch(t, tc.wantMatches, signals.MatchedEmbeddingRules)
			require.NotContains(t, signals.SignalConfidences, "embedding:below")
			require.Empty(t, signals.SignalErrors)
			if tc.wantRoute == "priority" {
				require.InDelta(t, 0.85, signals.ProjectionScores["evidence"], 0.001)
				require.Contains(t, signals.MatchedProjectionRules, "qualified")
			} else {
				require.Zero(t, signals.ProjectionScores["evidence"])
				require.NotContains(t, signals.SignalConfidences, "embedding:second")
			}
			result, err := c.EvaluateDecisionWithEngine(signals)
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, tc.wantRoute, result.Decision.Name)
		})
	}
}
