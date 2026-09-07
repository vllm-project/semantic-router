package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRecipeClassifiersIsolateMetadataSignals(t *testing.T) {
	denied := "denied"
	cfg := &config.RouterConfig{
		Recipes: []config.RoutingRecipe{
			{
				Name: "private",
				Profile: config.RoutingProfile{
					Signals: config.Signals{
						MetadataRules: []config.MetadataRule{{
							Name:      "consent-denied",
							Key:       "consent",
							Predicate: config.MetadataPredicate{Equals: &denied},
						}},
					},
					Decisions: []config.Decision{{
						Name: "metadata-route",
						Rules: config.RuleNode{
							Type: config.SignalTypeMetadata,
							Name: "consent-denied",
						},
					}},
				},
			},
			{
				Name: "public",
				Profile: config.RoutingProfile{
					Decisions: []config.Decision{{
						Name: "public-route",
						Rules: config.RuleNode{
							Type: config.SignalTypeKeyword,
							Name: "urgent",
						},
					}},
				},
			},
		},
	}
	classifiers, err := BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildRecipeClassifiers() error = %v", err)
	}

	input := SignalEvaluationInput{
		Text:            "hello",
		ContextText:     "hello",
		CurrentUserText: "hello",
		RequestFacts: RequestFacts{
			Metadata: map[string]string{"consent": "denied"},
		},
	}
	privateClassifier, ok := classifiers.ForRecipe("private")
	if !ok {
		t.Fatal("private recipe classifier is unavailable")
	}
	privateResults, err := privateClassifier.EvaluateAllSignalsWithHeaders(input)
	if err != nil {
		t.Fatalf("private EvaluateAllSignalsWithHeaders() error = %v", err)
	}
	if len(privateResults.MatchedMetadataRules) != 1 {
		t.Fatalf("private metadata signal did not match: %v", privateResults.MatchedMetadataRules)
	}

	publicClassifier, ok := classifiers.ForRecipe("public")
	if !ok {
		t.Fatal("public recipe classifier is unavailable")
	}
	publicResults, err := publicClassifier.EvaluateAllSignalsWithHeaders(input)
	if err != nil {
		t.Fatalf("public EvaluateAllSignalsWithHeaders() error = %v", err)
	}
	if len(publicResults.MatchedMetadataRules) != 0 {
		t.Fatalf(
			"metadata signals leaked across recipes: %v",
			publicResults.MatchedMetadataRules,
		)
	}
}
