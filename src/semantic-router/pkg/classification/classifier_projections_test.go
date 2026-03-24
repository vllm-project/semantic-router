package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestApplyProjectionsAddsDerivedOutputsAndScores(t *testing.T) {
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
								Weight:      0.6,
								ValueSource: "confidence",
							},
							{
								Type:   config.SignalTypeContext,
								Name:   "long_context",
								Weight: 0.2,
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
								LT:   float64Ptr(0.7),
							},
							{
								Name: "balance_reasoning",
								GTE:  float64Ptr(0.7),
							},
						},
					}},
				},
			},
		},
	}

	results := &SignalResults{
		MatchedKeywordRules: []string{"reasoning_request_markers"},
		MatchedContextRules: []string{"long_context"},
		SignalConfidences: map[string]float64{
			"keyword:reasoning_request_markers": 0.9,
		},
	}

	got := classifier.applyProjections(results)
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}
	if score := got.ProjectionScores["difficulty_score"]; score < 0.73 || score > 0.75 {
		t.Fatalf("difficulty_score = %.3f, want about 0.74", score)
	}
	if len(got.MatchedProjectionRules) != 1 || got.MatchedProjectionRules[0] != "balance_reasoning" {
		t.Fatalf("matched projection rules = %v, want [balance_reasoning]", got.MatchedProjectionRules)
	}
	confidence, ok := got.SignalConfidences["projection:balance_reasoning"]
	if !ok {
		t.Fatalf("projection confidence missing from %+v", got.SignalConfidences)
	}
	if confidence <= 0 || confidence > 1 {
		t.Fatalf("projection confidence = %.3f, want (0,1]", confidence)
	}
}

func TestApplyProjectionsInitializesSignalConfidenceMap(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "verification_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{{
							Type:   config.SignalTypeFactCheck,
							Name:   "needs_fact_check",
							Weight: 0.5,
						}},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "verification_band",
						Source: "verification_score",
						Method: "threshold_bands",
						Outputs: []config.ProjectionMappingOutput{{
							Name: "verification_required",
							GTE:  float64Ptr(0.3),
						}},
					}},
				},
			},
		},
	}

	got := classifier.applyProjections(&SignalResults{
		MatchedFactCheckRules: []string{"needs_fact_check"},
	})
	if got == nil || got.SignalConfidences == nil {
		t.Fatal("projection application did not initialize signal confidences")
	}
	if _, ok := got.SignalConfidences["projection:verification_required"]; !ok {
		t.Fatalf("projection confidence missing from %+v", got.SignalConfidences)
	}
}

func float64Ptr(v float64) *float64 {
	return &v
}
