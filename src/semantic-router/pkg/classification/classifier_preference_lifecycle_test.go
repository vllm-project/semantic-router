package classification

import (
	"errors"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestPreferenceLifecyclePreservesAuthoredPrototypeThresholdAndMargin(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	models.cfg.PreferenceRules = []config.PreferenceRule{
		{Name: "brief", Description: "Short responses", Examples: []string{"brief example"}, Threshold: .6},
		{Name: "detailed", Description: "Long responses", Examples: []string{"detailed example"}, Threshold: .6},
	}
	models.cfg.PreferenceModel = config.PreferenceModelConfig{PrototypeScoring: config.PrototypeScoringConfig{MarginThreshold: .05}}.WithDefaults()
	provider := newTestTextProvider(func(text string) ([]float32, error) {
		switch text {
		case "Short responses", "brief example", "brief request":
			return []float32{1, 0, 0}, nil
		case "Long responses", "detailed example", "detailed request":
			return []float32{0, 1, 0}, nil
		case "neutral request":
			return []float32{0, 0, 1}, nil
		case "ambiguous request":
			return []float32{1, 1, 0}, nil
		default:
			return nil, fmt.Errorf("unexpected embedding text %q", text)
		}
	})
	owner := &Classifier{Config: models.cfg, models: models, embeddingSet: embedding.NewSet(map[string]embedding.Provider{"mmbert": provider}, "mmbert")}
	if err := owner.initializePreferenceClassifier(); err != nil {
		t.Fatal(err)
	}
	if owner.preferenceClassifier == nil || owner.preferenceClassifier.judgment != nil || !owner.preferenceClassifier.useContrastive {
		t.Fatal("authored example bank selected native choice judgment")
	}
	for _, test := range []struct{ text, want string }{
		{"brief request", "brief"},
		{"detailed request", "detailed"},
		{"neutral request", ""},
		{"ambiguous request", ""},
	} {
		got, err := owner.preferenceClassifier.ClassifyContext(t.Context(), test.text)
		if test.want == "" {
			if !errors.Is(err, ErrPreferenceBelowThreshold) || got != nil {
				t.Fatalf("%s did not abstain: %+v %v", test.text, got, err)
			}
		} else if err != nil || got == nil || got.Preference != test.want {
			t.Fatalf("%s: %+v %v", test.text, got, err)
		}
	}
	if len(services.requests) != 0 || models.cfg.PreferenceModel.UseContrastive != nil {
		t.Fatal("prototype preparation invoked native judgment or mutated authored policy")
	}
}

func TestPreferenceLifecycleExplicitDecisionBindingOverridesPrototypes(t *testing.T) {
	models, services := preparedJudgmentModels(t)
	models.cfg.PreferenceRules = []config.PreferenceRule{
		{Name: "brief", Examples: []string{"Please be brief"}, Threshold: .6},
		{Name: "detailed", Examples: []string{"Please elaborate"}, Threshold: .6},
	}
	enabled := true
	models.cfg.PreferenceModel.UseContrastive = &enabled
	models.cfg.GlobalModelBindings = map[string]config.ModelBinding{"preference": {Deployment: "primary", Contract: config.DecisionTaskContract}}
	prepared, err := newClassifierModelRuntime(models.cfg, RecipeRuntimeOptions{Runtime: models.runtime})
	if err != nil {
		t.Fatal(err)
	}
	owner := &Classifier{Config: prepared.cfg, models: prepared}
	if err = owner.initializePreferenceClassifier(); err != nil {
		t.Fatal(err)
	}
	if owner.preferenceClassifier == nil || owner.preferenceClassifier.judgment == nil {
		t.Fatal("explicit decision task lost its native contract")
	}
	got, err := owner.preferenceClassifier.ClassifyContext(t.Context(), "Please be brief")
	if err != nil || got == nil || got.Preference != "brief" || len(services.requests) != 1 {
		t.Fatalf("explicit native preference: %+v %v", got, err)
	}
}
