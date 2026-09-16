package services

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestConvenienceDiagnosticsRecipeSelectionNeverFallsBack(t *testing.T) {
	cfg := &config.RouterConfig{Recipes: []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "private"}}}
	classifiers, setupErr := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if setupErr != nil {
		t.Fatal(setupErr)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	service := NewRecipeClassificationService(classifiers, cfg)
	t.Cleanup(func() { _ = service.Close() })
	selected, _ := classifiers.ForRecipe("private")
	selectedConfig, got, release, setupErr := service.AcquireRecipeRuntimeSnapshot("private")
	if setupErr != nil {
		t.Fatal(setupErr)
	}
	if got != selected || selectedConfig != selected.Config || selectedConfig.RoutingScope != "private" {
		t.Fatalf("selected recipe lost its classifier/config identity: %v", selectedConfig.RoutingScope)
	}
	release()
	defaultConfig, defaultClassifier, release, setupErr := service.AcquireRecipeRuntimeSnapshot("")
	if setupErr != nil {
		t.Fatal(setupErr)
	}
	if defaultConfig != cfg || defaultClassifier != classifiers.Default() {
		t.Fatal("omitted recipe changed legacy default")
	}
	release()
	view, release, setupErr := service.AcquireRecipeService("private")
	if setupErr != nil {
		t.Fatal(setupErr)
	}
	_, viewClassifier, viewRelease, setupErr := view.AcquireRecipeRuntimeSnapshot("private")
	if setupErr != nil || viewClassifier != selected {
		t.Fatalf("borrowed service view lost explicit scope: %v", setupErr)
	}
	viewRelease()
	release()

	checks := map[string]func(string) error{
		"pii": func(recipe string) error {
			_, err := service.DetectPII(context.Background(), PIIRequest{Recipe: recipe, Text: "hello"})
			return err
		},
		"guard": func(recipe string) error {
			_, err := service.CheckSecurity(context.Background(), SecurityRequest{Recipe: recipe, Text: "hello"})
			return err
		},
		"fact-check": func(recipe string) error {
			_, err := service.ClassifyFactCheck(context.Background(), FactCheckRequest{Recipe: recipe, Text: "hello"})
			return err
		},
		"feedback": func(recipe string) error {
			_, err := service.ClassifyUserFeedback(context.Background(), UserFeedbackRequest{Recipe: recipe, Text: "hello"})
			return err
		},
		"nli": func(recipe string) error {
			_, err := service.ClassifyNLI(context.Background(), NLIRequest{Recipe: recipe, Premise: "hello", Hypothesis: "a greeting"})
			return err
		},
	}
	for name, check := range checks {
		t.Run(name, func(t *testing.T) {
			if err := check("foreign"); !errors.Is(err, ErrUnknownDiagnosticRecipe) {
				t.Fatalf("unknown recipe fell back: %v", err)
			}
			if err := check("private"); !errors.Is(err, ErrClassifierUnavailable) {
				t.Fatalf("unloaded recipe must fail: %v", err)
			}
		})
	}
	fact, err := service.ClassifyFactCheck(context.Background(), FactCheckRequest{Text: "hello"})
	if err != nil || fact.Recipe != "default" || fact.ConfidenceAvailable {
		t.Fatalf("default placeholder/provenance changed: %+v, %v", fact, err)
	}
}
