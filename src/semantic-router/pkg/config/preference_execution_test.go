package config

import (
	"testing"

	"gopkg.in/yaml.v2"
)

func TestPreferenceExecutionSelectsConsistentModelConsumers(t *testing.T) {
	enabled, disabled := true, false
	for _, test := range []struct {
		name          string
		examples      bool
		contrastive   *bool
		model         string
		external      bool
		binding       string
		wantEmbedding bool
		wantTask      bool
	}{
		{name: "description only", wantTask: true},
		{name: "authored examples", examples: true, wantEmbedding: true},
		{name: "explicit embedding model", model: "qwen3", wantEmbedding: true},
		{name: "explicit prototypes", contrastive: &enabled, wantEmbedding: true},
		{name: "explicit native", examples: true, contrastive: &disabled, wantTask: true},
		{name: "external preserves examples", examples: true, external: true},
		{name: "explicit prototypes override external", external: true, contrastive: &enabled, wantEmbedding: true},
		{name: "global native binding", examples: true, contrastive: &enabled, binding: "global", wantTask: true},
		{name: "local native binding", examples: true, contrastive: &enabled, binding: "local", wantTask: true},
		{name: "native binding overrides external", external: true, binding: "local", wantTask: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.DecisionModel = "primary"
			cfg.PreferenceModel = PreferenceModelConfig{UseContrastive: test.contrastive, EmbeddingModel: test.model}.WithDefaults()
			cfg.PreferenceRules = []PreferenceRule{{Name: "brief", Description: "Short responses", Threshold: .6}}
			if test.examples {
				cfg.PreferenceRules[0].Examples = []string{"Please be brief"}
			}
			if test.external {
				cfg.ExternalModels = []ExternalModelConfig{{ModelRole: ModelRolePreference}}
			}
			binding := map[string]ModelBinding{"preference": {Deployment: "primary", Contract: DecisionTaskContract}}
			switch test.binding {
			case "global":
				cfg.GlobalModelBindings = binding
			case "local":
				cfg.ModelBindings = binding
			}
			cfg.Decisions = []Decision{{Rules: RuleCombination{Operator: "AND", Conditions: []RuleNode{{Type: SignalTypePreference, Name: "brief"}}}}}
			if got := cfg.PreferenceUsesPrototypes(); got != test.wantEmbedding {
				t.Fatalf("prototype selection=%v, want %v", got, test.wantEmbedding)
			}
			needed := EmbeddingModelsNeeded(cfg, "mmbert", false)
			model := test.model
			if model == "" {
				model = "mmbert"
			}
			if needed[model] != test.wantEmbedding || len(needed) > 1 {
				t.Fatalf("embedding demand=%v", needed)
			}
			if got := TaskConsumerInUse(cfg, DefaultRecipeName, "preference"); got != test.wantTask {
				t.Fatalf("native task demand=%v, want %v", got, test.wantTask)
			}
		})
	}
}

func TestPreferenceExecutionSurvivesCanonicalRoundTrip(t *testing.T) {
	for _, examples := range []bool{false, true} {
		source := `version: v0.3
providers:
  defaults:
    model: backend
  models:
    - name: backend
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: backend
  signals:
    preferences:
      - name: brief
        description: Short responses
        threshold: 0.6
`
		if examples {
			source += "        examples: [Please be brief]\n"
		}
		source += `  decisions:
    - name: brief-route
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: preference
            name: brief
      modelRefs:
        - model: backend
global:
  model_catalog:
    modules:
      classifier:
        preference:
          prototype_scoring:
            margin_threshold: 0.05
`
		cfg, err := ParseYAMLBytes([]byte(source))
		if err != nil {
			t.Fatal(err)
		}
		for generation := 0; generation < 3; generation++ {
			if cfg.PreferenceUsesPrototypes() != examples || cfg.PreferenceUsesDecisionTask() == examples {
				t.Fatalf("examples=%v, roundtrip=%d changed execution contract", examples, generation)
			}
			if cfg.PreferenceModel.UseContrastive != nil || cfg.PreferenceModel.PrototypeScoring.MarginThreshold != .05 {
				t.Fatal("preparation materialized a mode override or changed margin")
			}
			if got := EmbeddingModelsNeeded(cfg, "mmbert", false)["mmbert"]; got != examples {
				t.Fatalf("examples=%v, roundtrip=%d embedding demand=%v", examples, generation, got)
			}
			exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
			if err != nil {
				t.Fatal(err)
			}
			cfg, err = ParseYAMLBytes(exported)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}
