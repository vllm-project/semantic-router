package dsl

import (
	"testing"

	yamlv3 "gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func taxonomyConfigFixture() *config.RouterConfig {
	return &config.RouterConfig{
		TaxonomyClassifiers: []config.TaxonomyClassifierConfig{
			{
				Name: "privacy_classifier",
				Type: config.ClassifierTypeTaxonomy,
				Source: config.TaxonomyClassifierSource{
					Path:         "classifiers/privacy/",
					TaxonomyFile: "taxonomy.json",
				},
				Threshold:         0.55,
				SecurityThreshold: 0.7,
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				TaxonomyRules: []config.TaxonomySignalRule{
					{
						Name:       "privacy_policy",
						Classifier: "privacy_classifier",
						Bind: config.TaxonomySignalBind{
							Kind:  config.TaxonomyBindKindTier,
							Value: "privacy_policy",
						},
					},
				},
			},
			Projections: config.Projections{
				Scores: []config.ProjectionScore{
					{
						Name:   "privacy_contrastive_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{
							{
								Type:        config.ProjectionInputTaxonomyMetric,
								Classifier:  "privacy_classifier",
								Metric:      config.TaxonomyMetricContrastive,
								Weight:      1.0,
								ValueSource: "score",
							},
						},
					},
				},
			},
			Decisions: []config.Decision{
				{
					Name:      "local_privacy_policy",
					Priority:  250,
					ToolScope: "local_only",
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "taxonomy", Name: "privacy_policy"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "local/private-qwen"}},
				},
			},
		},
	}
}

func TestEmitYAMLFromConfigIncludesTaxonomyClassifierAndSignals(t *testing.T) {
	yamlBytes, err := EmitYAMLFromConfig(taxonomyConfigFixture())
	if err != nil {
		t.Fatalf("EmitYAMLFromConfig: %v", err)
	}

	var raw map[string]interface{}
	if err := yamlv3.Unmarshal(yamlBytes, &raw); err != nil {
		t.Fatalf("yaml.Unmarshal: %v", err)
	}

	global := mustMap(t, raw["global"], "global")
	modelCatalog := mustMap(t, global["model_catalog"], "global.model_catalog")
	classifiers := mustSlice(t, modelCatalog["classifiers"], "global.model_catalog.classifiers")
	if len(classifiers) != 1 {
		t.Fatalf("expected 1 classifier, got %d", len(classifiers))
	}

	routing := mustMap(t, raw["routing"], "routing")
	signals := mustMap(t, routing["signals"], "routing.signals")
	taxonomy := mustSlice(t, signals["taxonomy"], "routing.signals.taxonomy")
	if len(taxonomy) != 1 {
		t.Fatalf("expected 1 taxonomy signal, got %d", len(taxonomy))
	}
}

func TestDecompileRoutingToASTIncludesTaxonomyClassifierAndSignal(t *testing.T) {
	prog := DecompileRoutingToAST(taxonomyConfigFixture())

	if len(prog.Signals) == 0 {
		t.Fatal("expected at least one signal in AST")
	}
	found := false
	for _, sig := range prog.Signals {
		if sig.SignalType == config.SignalTypeTaxonomy && sig.Name == "privacy_policy" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("DecompileRoutingToAST should include taxonomy signal")
	}
}

func mustMap(t *testing.T, raw interface{}, path string) map[string]interface{} {
	t.Helper()
	typed, ok := raw.(map[string]interface{})
	if !ok {
		t.Fatalf("%s is not a map", path)
	}
	return typed
}

func mustSlice(t *testing.T, raw interface{}, path string) []interface{} {
	t.Helper()
	typed, ok := raw.([]interface{})
	if !ok {
		t.Fatalf("%s is not a slice", path)
	}
	return typed
}
