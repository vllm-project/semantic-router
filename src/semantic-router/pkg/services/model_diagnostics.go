package services

import (
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

var ErrUnknownDiagnosticRecipe = errors.New("diagnostic recipe is not available")

// diagnosticClassifierSnapshot is the shared scope selector for convenience
// diagnostics and explicit binding diagnostics. The caller holds runtimeMutex.
func (s *ClassificationService) diagnosticClassifierSnapshot(recipe string) (*classification.Classifier, *config.RouterConfig, string, error) {
	s.configMutex.RLock()
	defer s.configMutex.RUnlock()
	classifier := s.classifier
	scope := string(config.DefaultRecipeName)
	if classifier != nil && classifier.Config != nil && classifier.Config.RoutingScope != "" {
		scope = string(classifier.Config.RoutingScope)
	}
	if recipe != "" {
		if s.recipeClassifiers != nil {
			var ok bool
			classifier, ok = s.recipeClassifiers.ForRecipe(config.RecipeName(recipe))
			if !ok {
				return nil, nil, "", ErrUnknownDiagnosticRecipe
			}
		} else if recipe != scope {
			return nil, nil, "", ErrUnknownDiagnosticRecipe
		}
		scope = recipe
		if classifier == nil {
			return nil, nil, "", ErrClassifierUnavailable
		}
	}
	cfg := s.config
	if recipe != "" && classifier != nil {
		cfg = classifier.Config
	}
	return classifier, cfg, scope, nil
}

// AcquireRecipeRuntimeSnapshot leases one explicitly selected recipe. Empty
// recipe preserves convenience APIs' existing default scope. It never loads a
// model, and an unknown explicit recipe never falls back to the default.
func (s *ClassificationService) AcquireRecipeRuntimeSnapshot(recipe string) (*config.RouterConfig, *classification.Classifier, func(), error) {
	if s == nil {
		return nil, nil, func() {}, binding.ErrNotPrepared
	}
	s.runtimeMutex.RLock()
	var once sync.Once
	release := func() { once.Do(s.runtimeMutex.RUnlock) }
	if s.closed {
		release()
		return nil, nil, func() {}, binding.ErrClosed
	}
	classifier, cfg, _, err := s.diagnosticClassifierSnapshot(recipe)
	if err != nil {
		release()
		return nil, nil, func() {}, err
	}
	return cfg, classifier, release, nil
}

// AcquireRecipeService supplies a borrowed view for combined/batch operations.
// The view executes existing family adapters and cannot discover or own models.
func (s *ClassificationService) AcquireRecipeService(recipe string) (*ClassificationService, func(), error) {
	_, classifier, release, err := s.AcquireRecipeRuntimeSnapshot(recipe)
	if err != nil {
		return nil, func() {}, err
	}
	if classifier == nil {
		release()
		return nil, func() {}, ErrClassifierUnavailable
	}
	view := NewClassificationService(classifier, classifier.Config)
	return view, func() { _ = view.Close(); release() }, nil
}

// AcquireModelDiagnostics requires explicit scope. A normal Router caller also
// holds the live generation lease until the complete diagnostic operation ends.
func (s *ClassificationService) AcquireModelDiagnostics(recipe string) (ModelDiagnostics, func(), error) {
	if strings.TrimSpace(recipe) == "" {
		return ModelDiagnostics{}, func() {}, fmt.Errorf("%w: recipe is required", binding.ErrInvalidInput)
	}
	_, classifier, release, err := s.AcquireRecipeRuntimeSnapshot(recipe)
	if err != nil {
		return ModelDiagnostics{}, func() {}, err
	}
	diagnostics := ModelDiagnostics{Tasks: classifier.ModelDiagnosticRuntime()}
	if diagnostics.Tasks == nil {
		release()
		return ModelDiagnostics{}, func() {}, binding.ErrNotPrepared
	}
	return diagnostics, release, nil
}

// ModelDiagnostics holds one recipe's prepared bindings on its model runtime:
// classify, embedding and relevance bindings.
type ModelDiagnostics struct {
	Tasks *serving.Runtime
}

// PreparedBindings lists the runtime's ready bindings.
func (d ModelDiagnostics) PreparedBindings() []binding.PreparedBinding {
	return d.Tasks.PreparedBindings()
}
