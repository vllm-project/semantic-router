package config

import "testing"

func TestNeedsCoreMappingsForRouting(t *testing.T) {
	cfg := &RouterConfig{
		InlineModels: InlineModels{
			Classifier: Classifier{
				CategoryModel: CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
				},
				PIIModel: PIIModel{
					ModelID:        "models/mmbert32k-pii-detector-merged",
					PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
				},
			},
			PromptGuard: PromptGuardConfig{
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
		},
		IntelligentRouting: IntelligentRouting{
			Projections: Projections{
				Scores: []ProjectionScore{{
					Name:   "safety_score",
					Method: "weighted_sum",
					Inputs: []ProjectionScoreInput{{
						Type:   SignalTypeDomain,
						Name:   "billing",
						Weight: 1,
					}},
				}},
				Mappings: []ProjectionMapping{{
					Name:   "safety_band",
					Source: "safety_score",
					Method: "threshold_bands",
					Outputs: []ProjectionMappingOutput{{
						Name: "projection_domain_gate",
						GTE:  float64PtrForRoutingSignalUsageTest(0.5),
					}},
				}},
			},
		},
	}

	tests := []struct {
		name          string
		decisions     []Decision
		wantCategory  bool
		wantPII       bool
		wantJailbreak bool
	}{
		{
			name: "unused core signals",
			decisions: []Decision{{
				Name:  "default",
				Rules: RuleNode{Operator: "AND", Conditions: []RuleNode{}},
			}},
		},
		{
			name: "direct domain signal",
			decisions: []Decision{{
				Name:  "domain-route",
				Rules: RuleNode{Operator: "OR", Conditions: []RuleNode{{Type: SignalTypeDomain, Name: "billing"}}},
			}},
			wantCategory: true,
		},
		{
			name: "projection references domain signal",
			decisions: []Decision{{
				Name:  "projection-route",
				Rules: RuleNode{Operator: "OR", Conditions: []RuleNode{{Type: SignalTypeProjection, Name: "projection_domain_gate"}}},
			}},
			wantCategory: true,
		},
		{
			name: "direct pii and jailbreak signals",
			decisions: []Decision{{
				Name:  "safety-route",
				Rules: RuleNode{Operator: "OR", Conditions: []RuleNode{{Type: SignalTypePII, Name: "contains_pii"}, {Type: SignalTypeJailbreak, Name: "detector"}}},
			}},
			wantPII:       true,
			wantJailbreak: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg.Decisions = tt.decisions
			if got := cfg.NeedsCategoryMappingForRouting(); got != tt.wantCategory {
				t.Fatalf("NeedsCategoryMappingForRouting() = %v, want %v", got, tt.wantCategory)
			}
			if got := cfg.NeedsPIIMappingForRouting(); got != tt.wantPII {
				t.Fatalf("NeedsPIIMappingForRouting() = %v, want %v", got, tt.wantPII)
			}
			if got := cfg.NeedsJailbreakMappingForRouting(); got != tt.wantJailbreak {
				t.Fatalf("NeedsJailbreakMappingForRouting() = %v, want %v", got, tt.wantJailbreak)
			}
		})
	}
}

func float64PtrForRoutingSignalUsageTest(v float64) *float64 {
	return &v
}
