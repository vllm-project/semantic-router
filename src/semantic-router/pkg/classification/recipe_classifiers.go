package classification

import (
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RecipeClassifiers owns the immutable classifier graph for every routing
// recipe. Classifier instances are intentionally not shared between recipes:
// rule names are local, projection DAGs are local, and policy signals such as
// PII, jailbreak, and authz must only run when the selected recipe declares
// them. Heavy resources are shared through an explicitly owned model runtime.
type RecipeClassifiers struct {
	byRecipe     map[config.RecipeName]*Classifier
	order        []config.RecipeName
	runtimeOrder []config.RecipeName
	routingOrder []config.RecipeName
	// defaultRecipe names the recipe whose classifier serves the default APIs.
	defaultRecipe config.RecipeName
}

// RecipeRuntimeOptions borrows the generation's model runtime and default
// embedding snapshot. Named recipes prepare independent bindings against the
// same deployments and resource pool.
type RecipeRuntimeOptions struct {
	Runtime    *serving.Runtime
	Embeddings *embedding.Set
	// SystemOne asks the systemone deployments' remote APIs.
	SystemOne modelservice.Remote
}

// BuildRecipeClassifiers prepares the classifier graph and owned dependencies
// for each active recipe. InitializeRuntime completes family initialization
// before publication. Unreachable named recipes retain validated config only.
func BuildRecipeClassifiers(cfg *config.RouterConfig, runtimeOptions ...RecipeRuntimeOptions) (*RecipeClassifiers, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}

	options := RecipeRuntimeOptions{}
	if len(runtimeOptions) > 0 {
		options = runtimeOptions[0]
	}
	if options.Runtime == nil {
		options.Runtime = serving.New(nil, nil)
	}
	set := &RecipeClassifiers{
		byRecipe:      make(map[config.RecipeName]*Classifier),
		runtimeOrder:  make([]config.RecipeName, 0, len(cfg.Recipes)),
		routingOrder:  make([]config.RecipeName, 0, len(cfg.Recipes)),
		defaultRecipe: cfg.DefaultRecipe().Name,
	}
	if len(cfg.Recipes) == 0 {
		classifier, err := buildClassifier(cfg, options)
		if err != nil {
			return nil, fmt.Errorf("build routing recipe %q: %w", set.defaultRecipe, err)
		}
		set.byRecipe[set.defaultRecipe] = classifier
		set.order = append(set.order, set.defaultRecipe)
		set.runtimeOrder = append(set.runtimeOrder, set.defaultRecipe)
		if cfg.IsRecipeReachableForRouting(set.defaultRecipe) {
			set.routingOrder = append(set.routingOrder, set.defaultRecipe)
		}
		return set, nil
	}

	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		scopedConfig := cfg.ConfigForRecipe(recipe)
		reachable := cfg.IsRecipeReachableForRouting(recipe.Name)
		var classifier *Classifier
		var err error
		if recipe.Name != set.defaultRecipe && !reachable {
			// Canonical contract validation remains eager. Provisioning belongs
			// only to reachable routing scopes and the default API owner.
			err = config.ValidateKubernetesConfigContracts(scopedConfig)
			if err == nil {
				var models *classifierModelRuntime
				models, err = newClassifierModelRuntime(scopedConfig, options)
				if err == nil {
					classifier = &Classifier{Config: models.cfg, models: models}
				}
			}
		} else {
			classifier, err = buildClassifier(scopedConfig, options)
		}
		if err != nil {
			_ = set.Close()
			return nil, fmt.Errorf("build routing recipe %q: %w", recipe.Name, err)
		}
		set.byRecipe[recipe.Name] = classifier
		set.order = append(set.order, recipe.Name)
		if recipe.Name == set.defaultRecipe || reachable {
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
		// The default APIs of a default recipe that routes nothing own no
		// runtime: the decision model answers them.
		if recipeName == s.defaultRecipe && !s.routingRecipe(recipeName) {
			continue
		}
		if err := s.byRecipe[recipeName].InitializeRuntime(); err != nil {
			if closeErr := s.Close(); closeErr != nil {
				logging.ComponentWarnEvent("classifier", "recipe_runtime_initialization_rollback_failed", map[string]interface{}{
					"recipe": string(recipeName),
					"error":  closeErr.Error(),
				})
			}
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

// Close releases every recipe classifier's runtime resources.
func (s *RecipeClassifiers) Close() error {
	if s == nil {
		return nil
	}
	var errs []error
	for _, recipeName := range s.order {
		if err := s.byRecipe[recipeName].Close(); err != nil {
			errs = append(errs, fmt.Errorf("close routing recipe %q: %w", recipeName, err))
		}
	}
	return errors.Join(errs...)
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
	classifier, _ := s.ForRecipe(s.defaultRecipe)
	return classifier
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

// HasAnyPreparedEmbeddings reports actual providers across active recipe owners.
func (s *RecipeClassifiers) HasAnyPreparedEmbeddings() bool {
	return s.anyClassifierReady(func(c *Classifier) bool { return c.PreparedEmbeddings().Ready() })
}
