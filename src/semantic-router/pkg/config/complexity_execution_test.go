package config

import "testing"

func TestComplexityPrototypeDemandFollowsCandidatesAndExplicitBinding(t *testing.T) {
	for _, test := range []struct {
		name     string
		binding  bool
		text     bool
		image    bool
		wantText bool
		wantTask bool
	}{
		{name: "generic judgment", wantTask: true},
		{name: "text prototypes", text: true, wantText: true},
		{name: "explicit judgment", text: true, binding: true, wantTask: true},
		{name: "image prototypes", image: true, wantText: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.DecisionModel = "primary"
			rule := ComplexityRule{Name: "difficulty"}
			if test.text {
				rule.Hard.Candidates, rule.Easy.Candidates = []string{"hard example"}, []string{"easy example"}
			}
			if test.image {
				rule.Hard.ImageCandidates, rule.Easy.ImageCandidates = []string{"hard.png"}, []string{"easy.png"}
			}
			if test.binding {
				cfg.GlobalModelBindings = map[string]ModelBinding{"complexity": {Deployment: "primary", Contract: DecisionTaskContract}}
			}
			cfg.ComplexityRules = []ComplexityRule{rule}
			cfg.Decisions = []Decision{{Rules: RuleCombination{Operator: "AND", Conditions: []RuleNode{{Type: SignalTypeComplexity, Name: "difficulty:hard"}}}}}
			needed := EmbeddingModelsNeeded(cfg, "mmbert", false)
			if needed["mmbert"] != test.wantText || needed["multimodal"] != test.image {
				t.Fatalf("embedding demand=%v", needed)
			}
			if got := TaskConsumerInUse(cfg, DefaultRecipeName, "complexity"); got != test.wantTask {
				t.Fatalf("judgment demand=%v, want %v", got, test.wantTask)
			}
		})
	}
}

func TestComplexityPrototypeSelectionDoesNotInferBoundaryUnits(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.DecisionModel = "primary"
	for _, pair := range [][2]float64{{.025, -.08}, {.7, .3}} {
		rule := ComplexityRule{Name: "difficulty", HardAbove: &pair[0], EasyBelow: &pair[1]}
		if cfg.ComplexityRuleUsesPrototypes(rule) {
			t.Fatal("boundary numbers selected prototypes without authored examples")
		}
		rule.Hard.Candidates = []string{"hard example"}
		if !cfg.ComplexityRuleUsesPrototypes(rule) {
			t.Fatal("boundary numbers overrode authored examples")
		}
	}
}
