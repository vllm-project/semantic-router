package config

import "testing"

func TestUsesSignalTypeInRoutingTraversesProjectionDependencies(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Projections: Projections{
				Scores: []ProjectionScore{
					{
						Name:   "fact_score",
						Method: "weighted_sum",
						Inputs: []ProjectionScoreInput{{
							Type:   SignalTypeFactCheck,
							Name:   "verification-needed",
							Weight: 1,
						}},
					},
					{
						Name:   "combined_score",
						Method: "weighted_sum",
						Inputs: []ProjectionScoreInput{{
							Type:        SignalTypeProjection,
							Name:        "fact_score",
							ValueSource: ProjectionValueSourceScore,
							Weight:      1,
						}},
					},
					{
						Name:   "final_score",
						Method: "weighted_sum",
						Inputs: []ProjectionScoreInput{{
							Type:        SignalTypeProjection,
							Name:        "combined_high",
							ValueSource: ProjectionValueSourceConfidence,
							Weight:      1,
						}},
					},
				},
				Mappings: []ProjectionMapping{
					{
						Name:   "combined_band",
						Source: "combined_score",
						Outputs: []ProjectionMappingOutput{{
							Name: "combined_high",
							GTE:  float64PtrForRoutingSignalUsageTest(0.5),
						}},
					},
					{
						Name:   "final_band",
						Source: "final_score",
						Outputs: []ProjectionMappingOutput{{
							Name: "route_final",
							GTE:  float64PtrForRoutingSignalUsageTest(0.5),
						}},
					},
				},
			},
			Decisions: []Decision{{
				Name:  "projection-route",
				Rules: RuleNode{Type: SignalTypeProjection, Name: "route_final"},
			}},
		},
	}

	if !cfg.UsesSignalTypeInRouting(SignalTypeFactCheck) {
		t.Fatal("transitive projection chain did not retain its fact-check dependency")
	}
}

func TestUsesSignalTypeInRoutingProtectsAgainstProjectionCycles(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Projections: Projections{
				Scores: []ProjectionScore{
					{
						Name:   "cycle_a",
						Method: "weighted_sum",
						Inputs: []ProjectionScoreInput{{
							Type:   SignalTypeProjection,
							Name:   "cycle_b",
							Weight: 1,
						}},
					},
					{
						Name:   "cycle_b",
						Method: "weighted_sum",
						Inputs: []ProjectionScoreInput{{
							Type:   SignalTypeProjection,
							Name:   "cycle_a",
							Weight: 1,
						}},
					},
				},
				Mappings: []ProjectionMapping{{
					Name:   "cycle_band",
					Source: "cycle_a",
					Outputs: []ProjectionMappingOutput{{
						Name: "cycle_output",
						GTE:  float64PtrForRoutingSignalUsageTest(0.5),
					}},
				}},
			},
			Decisions: []Decision{{
				Name:  "cycle-route",
				Rules: RuleNode{Type: SignalTypeProjection, Name: "cycle_output"},
			}},
		},
	}

	if cfg.UsesSignalTypeInRouting(SignalTypeFactCheck) {
		t.Fatal("projection-only cycle unexpectedly reported a fact-check dependency")
	}
}

func TestReachableRoutingUsageExcludesUnmappedNamedRecipes(t *testing.T) {
	cfg := &RouterConfig{
		Recipes: []RoutingRecipe{
			{Name: DefaultRecipeName},
			{
				Name: "unmapped",
				Profile: RoutingProfile{Decisions: []Decision{{
					Name:  "named-route",
					Rules: RuleNode{Type: SignalTypeFactCheck, Name: "verification-needed"},
				}}},
			},
		},
	}

	if !cfg.UsesSignalTypeInRouting(SignalTypeFactCheck) {
		t.Fatal("whole-config routing inventory should retain declared recipe dependencies")
	}
	if cfg.UsesSignalTypeInReachableRouting(SignalTypeFactCheck) {
		t.Fatal("unmapped named recipe unexpectedly counted as request reachable")
	}

	cfg.Entrypoints = []EntrypointMapping{{
		ModelNames: []string{"vllm-sr/verification"},
		Recipe:     "unmapped",
	}}
	if !cfg.UsesSignalTypeInReachableRouting(SignalTypeFactCheck) {
		t.Fatal("entrypoint-mapped recipe did not count as request reachable")
	}
}

func float64PtrForRoutingSignalUsageTest(v float64) *float64 {
	return &v
}
