package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

type closeTrackingCache struct {
	closeCalls int
}

func (c *closeTrackingCache) IsEnabled() bool                       { return true }
func (c *closeTrackingCache) CheckConnection(context.Context) error { return nil }
func (c *closeTrackingCache) GetStats() cache.CacheStats            { return cache.CacheStats{} }
func (c *closeTrackingCache) AddEntry(context.Context, string, string, string, []byte, []byte, int) error {
	return nil
}

func (c *closeTrackingCache) LookupSimilarWithThreshold(context.Context, string, string, float32) (cache.LookupResult, error) {
	return cache.LookupResult{}, nil
}

func (c *closeTrackingCache) Close() error {
	c.closeCalls++
	return nil
}

func (c *closeTrackingCache) closeCount() int {
	return c.closeCalls
}

func TestOpenAIRouterCloseClosesOwnedResources(t *testing.T) {
	fakeCache := &closeTrackingCache{}
	router := &OpenAIRouter{Cache: fakeCache}

	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}

	if got := fakeCache.closeCount(); got != 1 {
		t.Fatalf("Cache.Close() called %d times, want 1", got)
	}
}

func TestOpenAIRouterCloseDelegatesToResourceScope(t *testing.T) {
	resourceClosers := 0
	resources := newResourceScope()
	resources.add(func() error {
		resourceClosers++
		return nil
	})

	fakeCache := &closeTrackingCache{}
	router := &OpenAIRouter{Cache: fakeCache, resources: resources}

	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}

	if resourceClosers != 1 {
		t.Fatalf("resource closers ran %d times, want exactly 1", resourceClosers)
	}
	if got := fakeCache.closeCount(); got != 0 {
		t.Fatalf("Close() bypassed the resource scope and closed Cache directly %d times", got)
	}
}

func TestCloseRecipeModelSelectorsClosesEveryRecipe(t *testing.T) {
	defaultRegistry := selection.NewRegistry()
	defaultSelector := &closeTrackingSelector{method: selection.MethodStatic}
	defaultRegistry.Register(selection.MethodStatic, defaultSelector)

	otherRegistry := selection.NewRegistry()
	otherSelector := &closeTrackingSelector{method: selection.MethodStatic}
	otherRegistry.Register(selection.MethodStatic, otherSelector)

	err := closeRecipeModelSelectors(map[config.RecipeName]*selection.Registry{
		config.DefaultRecipeName: defaultRegistry,
		"other":                  otherRegistry,
	})
	if err != nil {
		t.Fatalf("closeRecipeModelSelectors() error = %v", err)
	}

	if got := defaultSelector.closeCount(); got != 1 {
		t.Errorf("default recipe's selector closed %d times, want 1", got)
	}
	if got := otherSelector.closeCount(); got != 1 {
		t.Errorf("non-default recipe's selector closed %d times, want 1;"+
			" every recipe has its own registry", got)
	}
}

func TestCloseRecipeModelSelectorsDeduplicatesAliases(t *testing.T) {
	shared := selection.NewRegistry()
	selector := &closeTrackingSelector{method: selection.MethodStatic}
	shared.Register(selection.MethodStatic, selector)

	err := closeRecipeModelSelectors(map[config.RecipeName]*selection.Registry{
		config.DefaultRecipeName: shared,
		"alias":                  shared,
		"nil-entry":              nil,
	})
	if err != nil {
		t.Fatalf("closeRecipeModelSelectors() error = %v", err)
	}

	if got := selector.closeCount(); got != 1 {
		t.Fatalf("aliased registry's selector closed %d times, want exactly 1", got)
	}
}

type closeTrackingSelector struct {
	method selection.SelectionMethod
	closes int
}

func (s *closeTrackingSelector) Select(context.Context, *selection.SelectionContext) (*selection.SelectionResult, error) {
	return nil, nil
}

func (s *closeTrackingSelector) Method() selection.SelectionMethod { return s.method }

func (s *closeTrackingSelector) UpdateFeedback(context.Context, *selection.Feedback) error {
	return nil
}

func (s *closeTrackingSelector) Tier() selection.AlgorithmTier { return selection.TierSupported }

func (s *closeTrackingSelector) ExternalDependencies() []selection.Dependency { return nil }

func (s *closeTrackingSelector) Close() error {
	s.closes++
	return nil
}

func (s *closeTrackingSelector) closeCount() int {
	return s.closes
}
