package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildModelSpecsSkipsDisabledHallucinationFeatureModels(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-factcheck-classifier-merged": "llm-semantic-router/mmbert32k-factcheck-classifier-merged",
			"models/mom-halugate-detector":                 "llm-semantic-router/mom-halugate-detector",
			"models/mom-halugate-explainer":                "llm-semantic-router/mom-halugate-explainer",
		},
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				Enabled: false,
				FactCheckModel: config.FactCheckModelConfig{
					ModelID: "models/mmbert32k-factcheck-classifier-merged",
				},
				HallucinationModel: config.HallucinationModelConfig{
					ModelID: "models/mom-halugate-detector",
				},
				NLIModel: config.NLIModelConfig{
					ModelID: "models/mom-halugate-explainer",
				},
			},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 0", len(specs))
	}
}

func TestBuildModelSpecsIncludesFactCheckClassifierWhenSignalConfigured(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-factcheck-classifier-merged": "llm-semantic-router/mmbert32k-factcheck-classifier-merged",
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				FactCheckRules: []config.FactCheckRule{
					{Name: "needs_fact_check"},
				},
			},
		},
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				FactCheckModel: config.FactCheckModelConfig{
					ModelID: "models/mmbert32k-factcheck-classifier-merged",
				},
			},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 1", len(specs))
	}
	if specs[0].LocalPath != "models/mmbert32k-factcheck-classifier-merged" {
		t.Fatalf("LocalPath = %q, want fact-check classifier", specs[0].LocalPath)
	}
}

func TestBuildModelSpecsSkipsUnusedCoreClassifierModels(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-intent-classifier-merged":  "llm-semantic-router/mmbert32k-intent-classifier-merged",
			"models/mmbert32k-pii-detector-merged":       "llm-semantic-router/mmbert32k-pii-detector-merged",
			"models/mmbert32k-jailbreak-detector-merged": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
		},
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
				},
				PIIModel: config.PIIModel{
					ModelID:        "models/mmbert32k-pii-detector-merged",
					PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
				},
			},
			PromptGuard: config.PromptGuardConfig{
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "default-route",
				Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}},
			}},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 0", len(specs))
	}
}

func TestBuildModelSpecsIncludesUsedCoreClassifierModels(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-intent-classifier-merged":  "llm-semantic-router/mmbert32k-intent-classifier-merged",
			"models/mmbert32k-pii-detector-merged":       "llm-semantic-router/mmbert32k-pii-detector-merged",
			"models/mmbert32k-jailbreak-detector-merged": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
		},
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
				},
				PIIModel: config.PIIModel{
					ModelID:        "models/mmbert32k-pii-detector-merged",
					PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
				},
			},
			PromptGuard: config.PromptGuardConfig{
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name: "guarded-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					{Type: config.SignalTypeDomain, Name: "billing"},
					{Type: config.SignalTypePII, Name: "contains_pii"},
					{Type: config.SignalTypeJailbreak, Name: "detector"},
				}},
			}},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 3 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 3", len(specs))
	}
	assertContainsAllModelSpecs(t, specs,
		"models/mmbert32k-intent-classifier-merged",
		"models/mmbert32k-pii-detector-merged",
		"models/mmbert32k-jailbreak-detector-merged",
	)
}

func TestBuildModelSpecsIncludesCoreClassifierUsedViaProjection(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-jailbreak-detector-merged": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
		},
		InlineModels: config.InlineModels{
			PromptGuard: config.PromptGuardConfig{
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Projections: config.Projections{
				Scores: []config.ProjectionScore{{
					Name:   "risk_score",
					Method: "weighted_sum",
					Inputs: []config.ProjectionScoreInput{{Type: config.SignalTypeJailbreak, Name: "detector", Weight: 1.0}},
				}},
				Mappings: []config.ProjectionMapping{{
					Name:   "risk_map",
					Source: "risk_score",
					Method: "threshold",
					Outputs: []config.ProjectionMappingOutput{{
						Name: "high_risk",
					}},
				}},
			},
			Decisions: []config.Decision{{
				Name:  "guarded-route",
				Rules: config.RuleNode{Type: config.SignalTypeProjection, Name: "high_risk"},
			}},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 1", len(specs))
	}
	if specs[0].LocalPath != "models/mmbert32k-jailbreak-detector-merged" {
		t.Fatalf("LocalPath = %q, want jailbreak classifier", specs[0].LocalPath)
	}
}
