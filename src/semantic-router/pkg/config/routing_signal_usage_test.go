package config

import (
	"os"
	"path/filepath"
	"testing"
)

func newRoutingSignalUsageTestConfig() *RouterConfig {
	return &RouterConfig{
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
}

func TestNeedsCoreMappingsForRouting(t *testing.T) {
	cfg := newRoutingSignalUsageTestConfig()

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
		{
			name: "direct safety signals with mixed case and whitespace",
			decisions: []Decision{{
				Name:  "safety-route",
				Rules: RuleNode{Operator: "OR", Conditions: []RuleNode{{Type: " PII ", Name: "contains_pii"}, {Type: "JailBreak", Name: "detector"}}},
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

func TestNamedRecipeModelNeedsCoverProjectionsAndPlugins(t *testing.T) {
	cfg := loadGenericMultiRecipeModelNeedsTestConfig(t)

	if !cfg.NeedsFactCheckModelForRouting() {
		t.Fatal("fact-check model used through a named-recipe projection was not required")
	}
	if !cfg.NeedsFeedbackModelForRouting() {
		t.Fatal("feedback model used through a named-recipe projection was not required")
	}
	if !cfg.NeedsHallucinationDetectorForRouting() {
		t.Fatal("hallucination detector enabled by a named-recipe plugin was not required")
	}
	if !cfg.NeedsLocalHallucinationModelsForRouting() {
		t.Fatal("local hallucination models enabled by a named-recipe plugin were not required")
	}
	if !cfg.NeedsLocalHallucinationNLIForRouting() {
		t.Fatal("NLI requested by a named-recipe hallucination plugin was not required")
	}
}

func TestEndpointHallucinationBackendDoesNotNeedLocalModels(t *testing.T) {
	cfg := loadGenericMultiRecipeModelNeedsTestConfig(t)
	cfg.HallucinationMitigation.HallucinationModel.Backend = HallucinationBackendEndpoint

	if !cfg.NeedsHallucinationDetectorForRouting() {
		t.Fatal("endpoint hallucination detector should remain required by routing")
	}
	if cfg.NeedsLocalHallucinationModelsForRouting() {
		t.Fatal("endpoint hallucination detector must not require local model snapshots")
	}
	if cfg.NeedsLocalHallucinationNLIForRouting() {
		t.Fatal("endpoint hallucination detector must not require a local NLI snapshot")
	}
}

func TestDeclaredButUnusedModelSignalsDoNotRequireModels(t *testing.T) {
	cfg := &RouterConfig{
		InlineModels: InlineModels{
			HallucinationMitigation: HallucinationMitigationConfig{
				FactCheckModel: FactCheckModelConfig{ModelID: "models/test-fact-check"},
			},
			FeedbackDetector: FeedbackDetectorConfig{
				Enabled: true,
				ModelID: "models/test-feedback",
			},
		},
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				FactCheckRules:    []FactCheckRule{{Name: "verification-needed"}},
				UserFeedbackRules: []UserFeedbackRule{{Name: "correction-needed"}},
			},
			Decisions: []Decision{{
				Name:  "default-route",
				Rules: RuleNode{Operator: "AND", Conditions: []RuleNode{}},
			}},
		},
	}

	if cfg.NeedsFactCheckModelForRouting() {
		t.Fatal("declared but unused fact-check signal unexpectedly required a model")
	}
	if cfg.NeedsFeedbackModelForRouting() {
		t.Fatal("declared but unused feedback signal unexpectedly required a model")
	}
}

func TestNamedRecipeScopeDoesNotOwnDefaultAuxiliaryAPIs(t *testing.T) {
	cfg := &RouterConfig{
		RoutingScope: "named",
		InlineModels: InlineModels{
			HallucinationMitigation: HallucinationMitigationConfig{
				Enabled:            true,
				FactCheckModel:     FactCheckModelConfig{ModelID: "models/test-fact-check"},
				HallucinationModel: HallucinationModelConfig{ModelID: "models/test-hallucination"},
				NLIModel:           NLIModelConfig{ModelID: "models/test-nli"},
			},
			FeedbackDetector: FeedbackDetectorConfig{
				Enabled: true,
				ModelID: "models/test-feedback",
			},
		},
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				FactCheckRules:    []FactCheckRule{{Name: "verification-needed"}},
				UserFeedbackRules: []UserFeedbackRule{{Name: "correction-needed"}},
			},
		},
	}

	if cfg.NeedsFactCheckModelForAPI() ||
		cfg.NeedsFeedbackModelForAPI() ||
		cfg.NeedsHallucinationDetectorForDefaultRuntime() ||
		cfg.NeedsLocalHallucinationNLIForAPI() {
		t.Fatal("named recipe scope inherited default/API model consumers")
	}
}

func loadGenericMultiRecipeModelNeedsTestConfig(t *testing.T) *RouterConfig {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "generic-multi-recipe-model-needs.yaml"))
	if err != nil {
		t.Fatalf("read generic multi-recipe fixture: %v", err)
	}
	cfg, err := ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse generic multi-recipe fixture: %v", err)
	}
	return cfg
}

func float64PtrForRoutingSignalUsageTest(v float64) *float64 {
	return &v
}

func responseJailbreakPluginForRoutingSignalUsageTest(enabled bool) DecisionPlugin {
	return DecisionPlugin{
		Type: "response_jailbreak",
		Configuration: MustStructuredPayload(map[string]interface{}{
			"enabled": enabled,
			"action":  "block",
		}),
	}
}

func keywordDecisionsForRoutingSignalUsageTest(plugins ...DecisionPlugin) []Decision {
	return []Decision{{
		Name:    "keyword-route",
		Rules:   RuleNode{Operator: "OR", Conditions: []RuleNode{{Type: SignalTypeKeyword, Name: "probe"}}},
		Plugins: plugins,
	}}
}

// A response-direction rule is consumed by the selected decision's
// response_jailbreak plugin, never by a decision rule, so the mapping has to
// load on the rule (or on a plugin that classifies the response itself) alone.
func TestNeedsJailbreakMappingForResponseStageConsumers(t *testing.T) {
	responseRule := JailbreakRule{Name: "unsafe_completion", Direction: SignalDirectionResponse}

	tests := []struct {
		name        string
		promptGuard bool
		rules       []JailbreakRule
		decisions   []Decision
		want        bool
	}{
		{
			name:        "response-direction rule with no decision reading it",
			promptGuard: true,
			rules:       []JailbreakRule{responseRule},
			decisions:   keywordDecisionsForRoutingSignalUsageTest(),
			want:        true,
		},
		{
			name:        "response_jailbreak plugin classifies the response itself",
			promptGuard: true,
			decisions:   keywordDecisionsForRoutingSignalUsageTest(responseJailbreakPluginForRoutingSignalUsageTest(true)),
			want:        true,
		},
		{
			name:        "disabled response_jailbreak plugin",
			promptGuard: true,
			decisions:   keywordDecisionsForRoutingSignalUsageTest(responseJailbreakPluginForRoutingSignalUsageTest(false)),
			want:        false,
		},
		{
			name:        "request-direction rule nothing reads",
			promptGuard: true,
			rules:       []JailbreakRule{{Name: "prompt_injection"}},
			decisions:   keywordDecisionsForRoutingSignalUsageTest(),
			want:        false,
		},
		{
			name:        "prompt_guard disabled",
			promptGuard: false,
			rules:       []JailbreakRule{responseRule},
			decisions:   keywordDecisionsForRoutingSignalUsageTest(responseJailbreakPluginForRoutingSignalUsageTest(true)),
			want:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := newRoutingSignalUsageTestConfig()
			cfg.PromptGuard.Enabled = tt.promptGuard
			cfg.JailbreakRules = tt.rules
			cfg.Decisions = tt.decisions
			if got := cfg.NeedsJailbreakMappingForRouting(); got != tt.want {
				t.Fatalf("NeedsJailbreakMappingForRouting() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNeedsJailbreakMappingForResponseStageConsumersFollowsRecipeReachability(t *testing.T) {
	profile := func(rules []JailbreakRule, plugins ...DecisionPlugin) RoutingProfile {
		return RoutingProfile{
			Signals:   Signals{JailbreakRules: rules},
			Decisions: keywordDecisionsForRoutingSignalUsageTest(plugins...),
		}
	}
	responseRule := []JailbreakRule{{Name: "unsafe_completion", Direction: SignalDirectionResponse}}

	cfg := newRoutingSignalUsageTestConfig()
	cfg.Recipes = []RoutingRecipe{
		{Name: DefaultRecipeName, Profile: profile(nil)},
		{Name: "guarded", Profile: profile(responseRule)},
		{Name: "plugin-owned", Profile: profile(nil, responseJailbreakPluginForRoutingSignalUsageTest(true))},
	}
	if cfg.NeedsJailbreakMappingForRouting() {
		t.Fatal("response-stage consumers in recipes without an entrypoint must not load the jailbreak mapping")
	}

	cfg.Entrypoints = []EntrypointMapping{{ModelNames: []string{"vllm-sr/guarded"}, Recipe: "guarded"}}
	if !cfg.NeedsJailbreakMappingForRouting() {
		t.Fatal("a response-direction rule in a reachable recipe must load the jailbreak mapping")
	}

	cfg.Entrypoints = []EntrypointMapping{{ModelNames: []string{"vllm-sr/plugin-owned"}, Recipe: "plugin-owned"}}
	if !cfg.NeedsJailbreakMappingForRouting() {
		t.Fatal("a response_jailbreak plugin in a reachable recipe must load the jailbreak mapping")
	}

	for i, want := range []bool{false, true, true} {
		scoped := cfg.ConfigForRecipe(&cfg.Recipes[i])
		if got := scoped.NeedsJailbreakMappingForRouting(); got != want {
			t.Fatalf("recipe %q scoped NeedsJailbreakMappingForRouting() = %v, want %v", cfg.Recipes[i].Name, got, want)
		}
	}
}
