package dsl

import (
	"testing"

	yamlv3 "gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func taxonomyConfigFixture(t *testing.T) *config.RouterConfig {
	return &config.RouterConfig{
		KnowledgeBases: []config.KnowledgeBaseConfig{
			{
				Name: "privacy_kb",
				Source: config.KnowledgeBaseSource{
					Path:     "kb/privacy/",
					Manifest: "labels.json",
				},
				Threshold: 0.55,
				Groups: map[string][]string{
					"privacy_policy": {"proprietary_code"},
					"public":         {"generic_coding"},
				},
				Metrics: []config.KnowledgeBaseMetricConfig{
					{
						Name:          "private_vs_public",
						Type:          config.KBMetricTypeGroupMargin,
						PositiveGroup: "privacy_policy",
						NegativeGroup: "public",
					},
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				KBRules: []config.KBSignalRule{
					{
						Name: "privacy_policy",
						KB:   "privacy_kb",
						Target: config.KBSignalTarget{
							Kind:  config.KBTargetKindGroup,
							Value: "privacy_policy",
						},
						Match: config.KBMatchBest,
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
								Type:        config.ProjectionInputKBMetric,
								KB:          "privacy_kb",
								Metric:      "private_vs_public",
								Weight:      1.0,
								ValueSource: "score",
							},
						},
					},
				},
			},
			Decisions: []config.Decision{
				{
					Name:     "local_privacy_policy",
					Priority: 250,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleCondition{
							{Type: "kb", Name: "privacy_policy"},
						},
					},
					ModelRefs: []config.ModelRef{{Model: "local/private-qwen"}},
					Plugins: []config.DecisionPlugin{
						mustKBDecisionPlugin(t, config.DecisionPluginTools, config.ToolsPluginConfig{
							Enabled: true,
							Mode:    config.ToolsPluginModePassthrough,
						}),
					},
				},
			},
		},
	}
}

func TestEmitYAMLFromConfigIncludesKnowledgeBaseAndSignals(t *testing.T) {
	yamlBytes, err := EmitYAMLFromConfig(taxonomyConfigFixture(t))
	if err != nil {
		t.Fatalf("EmitYAMLFromConfig: %v", err)
	}

	var raw map[string]interface{}
	if err := yamlv3.Unmarshal(yamlBytes, &raw); err != nil {
		t.Fatalf("yaml.Unmarshal: %v", err)
	}

	global := mustMap(t, raw["global"], "global")
	modelCatalog := mustMap(t, global["model_catalog"], "global.model_catalog")
	kbs := mustSlice(t, modelCatalog["kbs"], "global.model_catalog.kbs")
	if len(kbs) != 1 {
		t.Fatalf("expected 1 knowledge base, got %d", len(kbs))
	}

	routing := mustMap(t, raw["routing"], "routing")
	signals := mustMap(t, routing["signals"], "routing.signals")
	kbSignals := mustSlice(t, signals["kb"], "routing.signals.kb")
	if len(kbSignals) != 1 {
		t.Fatalf("expected 1 kb signal, got %d", len(kbSignals))
	}
}

func TestDecompileRoutingToASTIncludesKnowledgeBaseSignal(t *testing.T) {
	prog := DecompileRoutingToAST(taxonomyConfigFixture(t))

	if len(prog.Signals) == 0 {
		t.Fatal("expected at least one signal in AST")
	}
	found := false
	for _, sig := range prog.Signals {
		if sig.SignalType == config.SignalTypeKB && sig.Name == "privacy_policy" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("DecompileRoutingToAST should include kb signal")
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

func mustKBDecisionPlugin(t *testing.T, pluginType string, cfg interface{}) config.DecisionPlugin {
	t.Helper()
	payload, err := config.NewStructuredPayload(cfg)
	if err != nil {
		t.Fatalf("NewStructuredPayload: %v", err)
	}
	return config.DecisionPlugin{Type: pluginType, Configuration: payload}
}
