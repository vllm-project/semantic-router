package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RecipeClassifiers owns the immutable classifier graph for every routing
// recipe. Classifier instances are intentionally not shared between recipes:
// rule names are local, projection DAGs are local, and policy signals such as
// PII, jailbreak, and authz must only run when the selected recipe declares
// them. Heavy model/provider resources remain shared through RouterConfig.
type RecipeClassifiers struct {
	byRecipe map[config.RecipeName]*Classifier
	order    []config.RecipeName
}

// BuildRecipeClassifiers compiles one classifier per normalized recipe without
// performing runtime initialization. Configs created programmatically without
// recipes use their single routing profile under the default name.
func BuildRecipeClassifiers(
	cfg *config.RouterConfig,
	categoryMapping *CategoryMapping,
	piiMapping *PIIMapping,
	jailbreakMapping *JailbreakMapping,
) (*RecipeClassifiers, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}

	set := &RecipeClassifiers{byRecipe: make(map[config.RecipeName]*Classifier)}
	if len(cfg.Recipes) == 0 {
		classifier, err := BuildClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
		if err != nil {
			return nil, fmt.Errorf("build routing recipe %q: %w", config.DefaultRecipeName, err)
		}
		set.byRecipe[config.DefaultRecipeName] = classifier
		set.order = append(set.order, config.DefaultRecipeName)
		return set, nil
	}

	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		scopedConfig := cfg.ConfigForRecipe(recipe)
		classifier, err := BuildClassifier(scopedConfig, categoryMapping, piiMapping, jailbreakMapping)
		if err != nil {
			return nil, fmt.Errorf("build routing recipe %q: %w", recipe.Name, err)
		}
		set.byRecipe[recipe.Name] = classifier
		set.order = append(set.order, recipe.Name)
	}
	return set, nil
}

// InitializeRuntime initializes every recipe classifier. Recipe context is
// preserved in errors so a startup failure points to the owning profile.
func (s *RecipeClassifiers) InitializeRuntime() error {
	if s == nil {
		return fmt.Errorf("recipe classifiers are nil")
	}
	for _, recipeName := range s.order {
		classifier := s.byRecipe[recipeName]
		if err := classifier.InitializeRuntime(); err != nil {
			return fmt.Errorf("initialize routing recipe %q: %w", recipeName, err)
		}
	}
	return nil
}

// ForRecipe returns the classifier for exactly one recipe. There is no
// implicit fallback because falling back would violate the isolation boundary.
func (s *RecipeClassifiers) ForRecipe(recipeName config.RecipeName) (*Classifier, bool) {
	if s == nil {
		return nil, false
	}
	classifier, ok := s.byRecipe[recipeName]
	return classifier, ok
}

// Default returns the classifier for the default recipe.
func (s *RecipeClassifiers) Default() *Classifier {
	classifier, _ := s.ForRecipe(config.DefaultRecipeName)
	return classifier
}
