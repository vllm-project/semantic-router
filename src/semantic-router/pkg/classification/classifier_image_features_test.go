package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestImageFeaturesRecipeDecisions(t *testing.T) {
	c := &Classifier{Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		Strategy: config.RoutingStrategyPriority,
		Signals:  config.Signals{InputModalityRules: []config.InputModalityRule{{Name: "has-image", Modality: config.InputModalityImage}}},
		Projections: config.Projections{
			Scores: []config.ProjectionScore{
				{Name: "blur_score", Inputs: []config.ProjectionScoreInput{{Type: "image_features", Name: "blur", Weight: 1, ValueSource: "raw"}}},
				{Name: "layout_score", Inputs: []config.ProjectionScoreInput{{Type: "image_features", Name: "edge_density", Weight: 1, ValueSource: "raw"}}},
			},
			Mappings: []config.ProjectionMapping{
				{Name: "blur_band", Source: "blur_score", Outputs: []config.ProjectionMappingOutput{{Name: "blurry", GTE: float64Ptr(.5)}}},
				{Name: "layout_band", Source: "layout_score", Outputs: []config.ProjectionMappingOutput{{Name: "dense", GTE: float64Ptr(.5)}, {Name: "simple", LT: float64Ptr(.5)}}},
			},
		},
		Decisions: []config.Decision{
			{Name: "strong-blur", Priority: 40, Rules: config.RuleNode{Type: "projection", Name: "blurry", OnUnknown: config.RuleOnUnknownNoMatch}},
			{Name: "strong-layout", Priority: 30, Rules: config.RuleNode{Type: "projection", Name: "dense", OnUnknown: config.RuleOnUnknownNoMatch}},
			{Name: "standard", Priority: 20, Rules: config.RuleNode{Type: "projection", Name: "simple", OnUnknown: config.RuleOnUnknownNoMatch}},
			{Name: "image-fallback", Priority: 10, Rules: config.RuleNode{Type: "input_modality", Name: "has-image"}},
		},
	}}}
	simple := encodeFeatureTestImage(t, func(x int) uint8 { return uint8(x / 128 * 255) })
	dense := encodeFeatureTestImage(t, func(x int) uint8 { return uint8(x % 2 * 255) })
	blurry := encodeFeatureTestImage(t, func(x int) uint8 { return uint8(x) })
	for _, tc := range []struct {
		name, want string
		images     []string
	}{
		{"simple", "standard", []string{simple}},
		{"dense", "strong-layout", []string{dense}},
		{"sharp and blurry", "strong-blur", []string{simple, blurry}},
		{"readable and unreadable", "standard", []string{"", simple}},
		{"none readable", "image-fallback", []string{"", "data:image/png;base64,broken"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			results := c.evaluateAllSignalsWithContext(SignalEvaluationInput{Text: "Describe this image", RequestFacts: RequestFacts{
				CurrentUserImages: tc.images, InputModality: InputModalityFacts{ImageContentCount: len(tc.images)},
			}}, nil, false)
			if tc.want == "image-fallback" {
				for _, key := range []string{"image_features:blur", "image_features:edge_density"} {
					if _, ok := results.SignalValues[key]; ok || results.SignalErrors[key] == "" {
						t.Fatalf("%s must remain unknown", key)
					}
				}
			}
			got, err := c.EvaluateDecisionWithEngine(results)
			if err != nil || got == nil || got.Decision.Name != tc.want {
				t.Fatalf("got %+v, %v; want %s", got, err, tc.want)
			}
		})
	}
	c.Config.Projections = config.Projections{}
	results := c.evaluateAllSignalsWithContext(SignalEvaluationInput{Text: "Describe this image", RequestFacts: RequestFacts{
		CurrentUserImages: []string{dense}, InputModality: InputModalityFacts{ImageContentCount: 1},
	}}, nil, false)
	got, err := c.EvaluateDecisionWithEngine(results)
	if err != nil || got == nil || got.Decision.Name != "image-fallback" {
		t.Fatalf("unconfigured recipe changed its image route: %+v, %v", got, err)
	}
}
