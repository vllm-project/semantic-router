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

func TestApplyMultiEmitAllMatchingBandsEmit(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "multi_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{{
							Type:        config.SignalTypeKeyword,
							Name:        "test_signal",
							Weight:      1.0,
							ValueSource: "confidence",
						}},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "multi_mapping",
						Source: "multi_score",
						Method: "multi_emit",
						Outputs: []config.ProjectionMappingOutput{
							{Name: "low_band", LT: float64Ptr(0.3)},
							{Name: "mid_band", GTE: float64Ptr(0.2), LT: float64Ptr(0.7)},
							{Name: "high_band", GTE: float64Ptr(0.5)},
						},
					}},
				},
			},
		},
	}

	results := &SignalResults{
		MatchedKeywordRules: []string{"test_signal"},
		SignalConfidences:   map[string]float64{"keyword:test_signal": 0.6},
	}

	got := classifier.applyProjections(results)
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}

	// Score = 0.6, should match mid_band and high_band
	matched := got.MatchedProjectionRules
	if len(matched) != 2 {
		t.Fatalf("expected 2 matched rules, got %d: %v", len(matched), matched)
	}
	if matched[0] != "mid_band" || matched[1] != "high_band" {
		t.Fatalf("expected [mid_band high_band], got %v", matched)
	}
}

func TestApplyTopKSelectsClosestBands(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "topk_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{{
							Type:        config.SignalTypeKeyword,
							Name:        "test_signal",
							Weight:      1.0,
							ValueSource: "confidence",
						}},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "topk_mapping",
						Source: "topk_score",
						Method: "top_k",
						TopK:   2,
						Outputs: []config.ProjectionMappingOutput{
							{Name: "band_0_4", GTE: float64Ptr(0.0), LT: float64Ptr(0.4)},  // center: 0.2
							{Name: "band_4_7", GTE: float64Ptr(0.4), LT: float64Ptr(0.7)},  // center: 0.55
							{Name: "band_7_1", GTE: float64Ptr(0.7)},                        // center: 0.7
						},
					}},
				},
			},
		},
	}

	results := &SignalResults{
		MatchedKeywordRules: []string{"test_signal"},
		SignalConfidences:   map[string]float64{"keyword:test_signal": 0.5},
	}

	got := classifier.applyProjections(results)
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}

	// Score = 0.5, only matches band_4_7 (since it's the only matching band)
	matched := got.MatchedProjectionRules
	if len(matched) != 1 {
		t.Fatalf("expected 1 matched rule, got %d: %v", len(matched), matched)
	}
	if matched[0] != "band_4_7" {
		t.Fatalf("expected [band_4_7], got %v", matched)
	}
}

func TestApplyTopKWithOverlappingBands(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "topk_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{{
							Type:        config.SignalTypeKeyword,
							Name:        "test_signal",
							Weight:      1.0,
							ValueSource: "confidence",
						}},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "topk_mapping",
						Source: "topk_score",
						Method: "top_k",
						TopK:   2,
						Outputs: []config.ProjectionMappingOutput{
							{Name: "narrow", GTE: float64Ptr(0.4), LT: float64Ptr(0.6)},  // center: 0.5
							{Name: "wide", GTE: float64Ptr(0.2), LT: float64Ptr(0.9)},    // center: 0.55
							{Name: "shifted", GTE: float64Ptr(0.3), LT: float64Ptr(0.8)}, // center: 0.55
						},
					}},
				},
			},
		},
	}

	// Score = 0.5: all three bands match.
	// |0.5 - 0.5| = 0.0 (narrow), |0.5 - 0.55| = 0.05 (wide, shifted)
	// Top 2 should be narrow first, then one of wide/shifted (stable order → wide).
	results := &SignalResults{
		MatchedKeywordRules: []string{"test_signal"},
		SignalConfidences:   map[string]float64{"keyword:test_signal": 0.5},
	}

	got := classifier.applyProjections(results)
	matched := got.MatchedProjectionRules
	if len(matched) != 2 {
		t.Fatalf("expected 2 matched rules, got %d: %v", len(matched), matched)
	}
	if matched[0] != "narrow" {
		t.Fatalf("expected narrow as closest band, got %v", matched)
	}
}

