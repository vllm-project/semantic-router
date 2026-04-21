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

func TestProjectionInputValueRawReadsSignalValues(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "workload_pressure",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{
							{
								Type:        "structure",
								Name:        "many_questions",
								Weight:      0.5,
								ValueSource: "raw",
							},
							{
								Type:        "conversation",
								Name:        "tool_loop_deep",
								Weight:      0.3,
								ValueSource: "raw",
							},
						},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "pressure_band",
						Source: "workload_pressure",
						Method: "threshold_bands",
						Outputs: []config.ProjectionMappingOutput{
							{Name: "low_pressure", LT: float64Ptr(2.0)},
							{Name: "high_pressure", GTE: float64Ptr(2.0)},
						},
					}},
				},
			},
		},
	}

	results := &SignalResults{
		SignalValues: map[string]float64{
			"structure:many_questions":    4.0,
			"conversation:tool_loop_deep": 7.0,
		},
	}

	got := classifier.applyProjections(results)
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}
	// 0.5*4.0 + 0.3*7.0 = 2.0 + 2.1 = 4.1
	score := got.ProjectionScores["workload_pressure"]
	if score < 4.09 || score > 4.11 {
		t.Fatalf("workload_pressure = %.3f, want about 4.1", score)
	}
	if len(got.MatchedProjectionRules) != 1 || got.MatchedProjectionRules[0] != "high_pressure" {
		t.Fatalf("matched projection rules = %v, want [high_pressure]", got.MatchedProjectionRules)
	}
}

func TestProjectionInputValueRawReturnsZeroWhenMissing(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "absent_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{{
							Type:        "structure",
							Name:        "nonexistent_signal",
							Weight:      1.0,
							ValueSource: "raw",
						}},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "absent_band",
						Source: "absent_score",
						Method: "threshold_bands",
						Outputs: []config.ProjectionMappingOutput{
							{Name: "zero_output", LT: float64Ptr(0.01)},
						},
					}},
				},
			},
		},
	}

	got := classifier.applyProjections(&SignalResults{
		SignalValues: map[string]float64{},
	})
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}
	if score := got.ProjectionScores["absent_score"]; score != 0 {
		t.Fatalf("absent_score = %.3f, want 0", score)
	}
}

func TestProjectionInputValueRawReturnsZeroWhenNilMap(t *testing.T) {
	result := projectionInputValue(config.ProjectionScoreInput{
		Type:        "structure",
		Name:        "any",
		Weight:      1.0,
		ValueSource: "raw",
	}, &SignalResults{SignalValues: nil})

	if result != 0 {
		t.Fatalf("expected 0 for nil SignalValues, got %.3f", result)
	}
}

func TestProjectionMixedRawAndConfidenceInputs(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Projections: config.Projections{
					Scores: []config.ProjectionScore{{
						Name:   "mixed_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{
							{
								Type:        "structure",
								Name:        "many_questions",
								Weight:      0.4,
								ValueSource: "raw",
							},
							{
								Type:        config.SignalTypeKeyword,
								Name:        "reasoning_markers",
								Weight:      0.6,
								ValueSource: "confidence",
							},
						},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "mixed_band",
						Source: "mixed_score",
						Method: "threshold_bands",
						Outputs: []config.ProjectionMappingOutput{
							{Name: "low", LT: float64Ptr(1.5)},
							{Name: "high", GTE: float64Ptr(1.5)},
						},
					}},
				},
			},
		},
	}

	results := &SignalResults{
		SignalValues: map[string]float64{
			"structure:many_questions": 3.0,
		},
		MatchedKeywordRules: []string{"reasoning_markers"},
		SignalConfidences: map[string]float64{
			"keyword:reasoning_markers": 0.85,
		},
	}

	got := classifier.applyProjections(results)
	if got == nil {
		t.Fatal("applyProjections returned nil")
	}
	// 0.4*3.0 + 0.6*0.85 = 1.2 + 0.51 = 1.71
	score := got.ProjectionScores["mixed_score"]
	if score < 1.70 || score > 1.72 {
		t.Fatalf("mixed_score = %.4f, want about 1.71", score)
	}
	if len(got.MatchedProjectionRules) != 1 || got.MatchedProjectionRules[0] != "high" {
		t.Fatalf("matched = %v, want [high]", got.MatchedProjectionRules)
	}
}

func TestProjectionInputValueRawNegativeValues(t *testing.T) {
	result := projectionInputValue(config.ProjectionScoreInput{
		Type:        "structure",
		Name:        "neg_signal",
		Weight:      1.0,
		ValueSource: "raw",
	}, &SignalResults{
		SignalValues: map[string]float64{
			"structure:neg_signal": -2.5,
		},
	})

	if result != -2.5 {
		t.Fatalf("expected -2.5 for negative raw value, got %.3f", result)
	}
}

func TestProjectionInputValueRawCaseInsensitive(t *testing.T) {
	for _, vs := range []string{"Raw", "RAW", " raw ", " Raw"} {
		result := projectionInputValue(config.ProjectionScoreInput{
			Type:        "Structure",
			Name:        "sig",
			Weight:      1.0,
			ValueSource: vs,
		}, &SignalResults{
			SignalValues: map[string]float64{
				"structure:sig": 5.0,
			},
		})
		if result != 5.0 {
			t.Fatalf("value_source=%q: expected 5.0, got %.3f", vs, result)
		}
	}
}

func TestProjectionInputValueRawExplicitZero(t *testing.T) {
	result := projectionInputValue(config.ProjectionScoreInput{
		Type:        "structure",
		Name:        "zero_signal",
		Weight:      1.0,
		ValueSource: "raw",
	}, &SignalResults{
		SignalValues: map[string]float64{
			"structure:zero_signal": 0.0,
		},
	})

	if result != 0 {
		t.Fatalf("expected 0 for explicit zero raw value, got %.3f", result)
	}
}

func TestProjectionInputValueRawLargeValues(t *testing.T) {
	result := projectionInputValue(config.ProjectionScoreInput{
		Type:        "structure",
		Name:        "token_count",
		Weight:      1.0,
		ValueSource: "raw",
	}, &SignalResults{
		SignalValues: map[string]float64{
			"structure:token_count": 128000.0,
		},
	})

	if result != 128000.0 {
		t.Fatalf("expected 128000 for large raw value, got %.1f", result)
	}
}

func float64Ptr(v float64) *float64 {
	return &v
}
