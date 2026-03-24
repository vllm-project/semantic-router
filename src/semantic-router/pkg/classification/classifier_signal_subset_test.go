package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRefreshSignalFamiliesRecomputesProjections(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "difficulty_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{
							{
								Type:        config.SignalTypeKeyword,
								Name:        "reasoning_request_markers",
								Weight:      0.4,
								ValueSource: "confidence",
							},
							{
								Type:        config.SignalTypeEmbedding,
								Name:        "semantic_match",
								Weight:      0.6,
								ValueSource: "confidence",
							},
						},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "difficulty_band",
						Source: "difficulty_score",
						Method: "threshold_bands",
						Outputs: []config.ProjectionMappingOutput{
							{
								Name: "balance_medium",
								LT:   subsetFloat64Ptr(0.7),
							},
							{
								Name: "balance_reasoning",
								GTE:  subsetFloat64Ptr(0.7),
							},
						},
					}},
				},
			},
		},
	}

	base := &SignalResults{
		MatchedKeywordRules:    []string{"reasoning_request_markers"},
		MatchedEmbeddingRules:  []string{"semantic_match"},
		MatchedProjectionRules: []string{"balance_medium"},
		SignalConfidences: map[string]float64{
			"keyword:reasoning_request_markers": 0.95,
			"embedding:semantic_match":          0.10,
			"projection:balance_medium":         0.42,
		},
		ProjectionScores: map[string]float64{
			"difficulty_score": 0.43,
		},
		ProjectionBoundaryDistances: map[string]float64{
			"balance_medium": 0.27,
		},
		Metrics: &SignalMetricsCollection{},
	}
	refreshed := &SignalResults{
		MatchedEmbeddingRules: []string{"semantic_match"},
		SignalConfidences: map[string]float64{
			"embedding:semantic_match": 0.95,
		},
		Metrics: &SignalMetricsCollection{
			Embedding: SignalMetrics{Confidence: 0.95},
		},
	}

	got := classifier.RefreshSignalFamilies(base, refreshed, []string{config.SignalTypeEmbedding})
	if got == nil {
		t.Fatal("RefreshSignalFamilies returned nil")
	}
	if score := got.ProjectionScores["difficulty_score"]; score <= 0.9 {
		t.Fatalf("difficulty_score = %.3f, want > 0.9 after refresh", score)
	}
	if len(got.MatchedProjectionRules) != 1 || got.MatchedProjectionRules[0] != "balance_reasoning" {
		t.Fatalf("matched projection rules = %v, want [balance_reasoning]", got.MatchedProjectionRules)
	}
	if confidence := got.SignalConfidences["embedding:semantic_match"]; confidence != 0.95 {
		t.Fatalf("embedding confidence = %.2f, want 0.95", confidence)
	}
	if confidence := got.SignalConfidences["projection:balance_reasoning"]; confidence <= 0 {
		t.Fatalf("projection confidence = %.2f, want > 0", confidence)
	}
}

func subsetFloat64Ptr(v float64) *float64 {
	return &v
}

func TestRefreshSignalFamiliesReplacesPIIMetadata(t *testing.T) {
	classifier := &Classifier{Config: &config.RouterConfig{}}
	base := &SignalResults{
		PIIDetected:     true,
		PIIEntities:     []string{"EMAIL"},
		MatchedPIIRules: []string{"email"},
		Metrics:         &SignalMetricsCollection{},
	}
	refreshed := &SignalResults{
		PIIDetected: false,
		Metrics:     &SignalMetricsCollection{},
	}

	got := classifier.RefreshSignalFamilies(base, refreshed, []string{config.SignalTypePII})
	if got.PIIDetected {
		t.Fatal("expected refreshed PII detection to clear previous true value")
	}
	if len(got.PIIEntities) != 0 {
		t.Fatalf("pii entities = %v, want empty", got.PIIEntities)
	}
	if len(got.MatchedPIIRules) != 0 {
		t.Fatalf("matched pii rules = %v, want empty", got.MatchedPIIRules)
	}
}
