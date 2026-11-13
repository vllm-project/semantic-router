package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildRouterOwnsPreparedEmbeddings(t *testing.T) {
	components, err := buildRouterComponents(&config.RouterConfig{})
	require.NoError(t, err)
	router := components.buildRouter()
	defer func() { require.NoError(t, router.Close()) }()

	require.NotNil(t, components.embeddings)
	require.Same(t, components.embeddings, router.Embeddings)
}

// The embedding deployment starts only for a recall consumer: a response
// cache no decision uses, or one that reuses exact answers only, embeds
// nothing, and neither does a configuration with no recall at all.
func TestBuildRouterDoesNotLoadEmbeddingForIdleOrExactOnlyCache(t *testing.T) {
	for _, mode := range []string{"no recall", "unused", "exact"} {
		t.Run(mode, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.SemanticCache.InUse = mode != "no recall"
			cfg.EmbeddingDeployment = "unused-cache"
			cfg.ModelDeployments = map[string]config.ModelDeployment{"unused-cache": {Provider: config.ModelRuntimeProvider, Artifact: "/not-installed/unused-cache-embedding"}}
			cfg.Decisions = []config.Decision{{Name: "route"}}
			if mode == "exact" {
				cfg.Decisions[0].Plugins = []config.DecisionPlugin{{Type: "response_cache", Configuration: config.MustStructuredPayload(config.ResponseCachePluginConfig{Enabled: true, Mode: "exact"})}}
			}
			components, err := buildRouterComponents(cfg)
			require.NoError(t, err)
			router := components.buildRouter()
			t.Cleanup(func() { require.NoError(t, router.Close()) })
			require.False(t, components.embeddings.Ready())
			require.False(t, components.serviceEmbeddings.Ready())
			require.Empty(t, components.serving.PreparedBindings())
			require.Equal(t, mode == "exact", components.semanticCache.IsEnabled())
			require.Empty(t, components.semanticCacheIdentity)
		})
	}
}
