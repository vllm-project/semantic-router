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
	byRecipe     map[config.RecipeName]*Classifier
	order        []config.RecipeName
	runtimeOrder []config.RecipeName
	routingOrder []config.RecipeName
}

// BuildRecipeClassifiers compiles one classifier per explicit Recipe without
// performing runtime initialization. An empty managed snapshot produces an
// empty set; it never creates a classifier from root-level scoped fields.
func BuildRecipeClassifiers(
	cfg *config.RouterConfig,
	categoryMapping *CategoryMapping,
	piiMapping *PIIMapping,
	jailbreakMapping *JailbreakMapping,
) (*RecipeClassifiers, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}

	set := &RecipeClassifiers{
		byRecipe:     make(map[config.RecipeName]*Classifier),
		runtimeOrder: make([]config.RecipeName, 0, len(cfg.Recipes)),
		routingOrder: make([]config.RecipeName, 0, len(cfg.Recipes)),
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
		reachable := cfg.IsRecipeReachableForRouting(recipe.Name)
		if recipe.Name == config.DefaultRecipeName || reachable {
			set.runtimeOrder = append(set.runtimeOrder, recipe.Name)
		}
		if reachable {
			set.routingOrder = append(set.routingOrder, recipe.Name)
		}
	}
	return set, nil
}

// InitializeRuntime initializes the default/API classifier and every
// request-reachable named recipe. All declared recipes were still built above,
// preserving construction and validation semantics without provisioning
// unreachable runtime resources.
func (s *RecipeClassifiers) InitializeRuntime() error {
	if s == nil {
		return fmt.Errorf("recipe classifiers are nil")
	}
	for _, recipeName := range s.lifecycleOrder() {
		classifier := s.byRecipe[recipeName]
		var err error
		if recipeName == config.DefaultRecipeName && !s.routingRecipe(recipeName) {
			err = classifier.InitializeDefaultAPIRuntime()
		} else {
			err = classifier.InitializeRuntime()
		}
		if err != nil {
			return fmt.Errorf("initialize routing recipe %q: %w", recipeName, err)
		}
	}
	return nil
}

func (s *RecipeClassifiers) lifecycleOrder() []config.RecipeName {
	if s == nil {
		return nil
	}
	if s.runtimeOrder != nil {
		return s.runtimeOrder
	}
	return s.order
}

func (s *RecipeClassifiers) routingRecipe(recipeName config.RecipeName) bool {
	for _, candidate := range s.routingLifecycleOrder() {
		if candidate == recipeName {
			return true
		}
	}
	return false
}

func (s *RecipeClassifiers) routingLifecycleOrder() []config.RecipeName {
	if s == nil {
		return nil
	}
	if s.routingOrder != nil {
		return s.routingOrder
	}
	return s.order
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

// HasFactCheckClassifier reports default/API fact-check readiness.
func (s *RecipeClassifiers) HasFactCheckClassifier() bool {
	return factCheckClassifierReady(s.Default())
}

// HasHallucinationDetector reports default/API detector readiness.
func (s *RecipeClassifiers) HasHallucinationDetector() bool {
	classifier := s.Default()
	return classifier != nil && classifier.IsHallucinationDetectorReady()
}

// HasHallucinationExplainer reports default/API NLI readiness.
func (s *RecipeClassifiers) HasHallucinationExplainer() bool {
	classifier := s.Default()
	return classifier != nil && classifier.IsHallucinationExplainerReady()
}

// HasFeedbackDetector reports default/API feedback readiness.
func (s *RecipeClassifiers) HasFeedbackDetector() bool {
	return feedbackDetectorReady(s.Default())
}

// HasAnyFactCheckClassifier reports aggregate reachable-recipe inventory
// readiness without changing default API readiness semantics.
func (s *RecipeClassifiers) HasAnyFactCheckClassifier() bool {
	return s.anyClassifierReady(factCheckClassifierReady)
}

// HasAnyHallucinationDetector reports aggregate reachable-recipe detector
// inventory readiness.
func (s *RecipeClassifiers) HasAnyHallucinationDetector() bool {
	return s.anyClassifierReady((*Classifier).IsHallucinationDetectorReady)
}

// HasAnyHallucinationExplainer reports aggregate reachable-recipe NLI
// inventory readiness.
func (s *RecipeClassifiers) HasAnyHallucinationExplainer() bool {
	return s.anyClassifierReady((*Classifier).IsHallucinationExplainerReady)
}

// HasAnyFeedbackDetector reports aggregate reachable-recipe feedback inventory
// readiness.
func (s *RecipeClassifiers) HasAnyFeedbackDetector() bool {
	return s.anyClassifierReady(feedbackDetectorReady)
}

func factCheckClassifierReady(classifier *Classifier) bool {
	return classifier != nil &&
		classifier.factCheckClassifier != nil &&
		classifier.factCheckClassifier.IsInitialized()
}

func feedbackDetectorReady(classifier *Classifier) bool {
	return classifier != nil &&
		classifier.feedbackDetector != nil &&
		classifier.feedbackDetector.IsInitialized()
}

func (s *RecipeClassifiers) anyClassifierReady(ready func(*Classifier) bool) bool {
	if s == nil {
		return false
	}
	for _, recipeName := range s.lifecycleOrder() {
		classifier := s.byRecipe[recipeName]
		if classifier != nil && ready(classifier) {
			return true
		}
	}
	return false
}

// PreloadKnowledgeBases prepares recipe-local KB classifiers without sharing
// policy or symbol state across recipe boundaries.
func (s *RecipeClassifiers) PreloadKnowledgeBases() error {
	if s == nil {
		return nil
	}
	for _, recipeName := range s.routingLifecycleOrder() {
		if err := s.byRecipe[recipeName].PreloadKnowledgeBases(); err != nil {
			return fmt.Errorf("preload routing recipe %q: %w", recipeName, err)
		}
	}
	return nil
}
