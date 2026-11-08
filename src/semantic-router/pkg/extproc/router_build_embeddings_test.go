package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

func TestBuildRouterDoesNotLoadEmbeddingForIdleOrExactOnlyCache(t *testing.T) {
	for _, mode := range []string{"unused", "exact"} {
		t.Run(mode, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.SemanticCache.Enabled = true
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
			require.Empty(t, components.serving.PreparedBindings())
			require.Equal(t, mode == "exact", components.semanticCache.IsEnabled())
			require.Empty(t, components.semanticCacheIdentity)
		})
	}
}

func TestBuildRouterOwnsGlobalEmbeddingAPIWithoutRoutingDemand(t *testing.T) {
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var input struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			t.Error(err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		data := make([]map[string]any, len(input.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": []float32{1, 0}}
		}
		if err := json.NewEncoder(w).Encode(map[string]any{"data": data}); err != nil {
			t.Error(err)
		}
	}))
	defer endpoint.Close()
	cfg := &config.RouterConfig{}
	cfg.API.Embeddings.Enabled = true
	cfg.EmbeddingDeployment = "global"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {
		Provider: config.OpenAIEmbeddingsProvider, Endpoint: endpoint.URL, Model: "global", Capabilities: &config.DeploymentCapabilities{Dimensions: 2},
	}}
	components, err := buildRouterComponents(cfg)
	require.NoError(t, err)
	router := components.buildRouter()
	defer func() { require.NoError(t, router.Close()) }()
	require.False(t, components.embeddings.Ready())
	scope, prepared, release, err := components.classificationSvc.AcquireEmbeddingAPISnapshot()
	defer release()
	require.NoError(t, err)
	require.Equal(t, config.GlobalModelScope, scope.RoutingScope)
	require.Same(t, components.serviceEmbeddings, prepared)
	provider, err := prepared.Default()
	require.NoError(t, err)
	vector, err := provider.Embed(context.Background(), "hello")
	require.NoError(t, err)
	require.Equal(t, []float32{1, 0}, vector)
	bindings := components.serving.PreparedBindings()
	require.Len(t, bindings, 1)
	require.Equal(t, "api.embedding", bindings[0].Identity.Name)
}
