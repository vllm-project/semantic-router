package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestBuildEarlyResourcesDefersSemanticCacheUntilClassifierRuntimeReady(t *testing.T) {
	// Keep this regression independent of a native model or a Milvus process.
	// The ordering is the lifecycle contract that prevents a cold-start cache
	// constructor from querying the native embedding contract too early.
	var order []string
	components := &routerComponents{
		cfg:       &config.RouterConfig{},
		resources: newResourceScope(),
	}
	t.Cleanup(func() { require.NoError(t, components.resources.close()) })

	err := components.buildEarlyResourcesWith(
		&classifierMappings{},
		func(
			cfg *config.RouterConfig,
			mappings *classifierMappings,
		) (*classification.RecipeClassifiers, *classification.Classifier, *services.ClassificationService, error) {
			classifiers, classifier, service, err := createRouterClassifier(cfg, mappings)
			if err == nil {
				order = append(order, "classifier_runtime")
			}
			return classifiers, classifier, service, err
		},
		func(*config.RouterConfig) (cache.CacheBackend, error) {
			order = append(order, "semantic_cache")
			return cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: true}), nil
		},
	)
	require.NoError(t, err)
	require.Equal(t, []string{"classifier_runtime", "semantic_cache"}, order)
	require.NotNil(t, components.classifier)
	require.NotNil(t, components.semanticCache)
}
