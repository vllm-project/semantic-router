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
			cfg.SemanticCache.EmbeddingModel = "mmbert"
			cfg.MmBertModelPath = "/not-installed/unused-cache-embedding"
			cfg.SemanticCache.PolarityGuard = &config.PolarityGuardConfig{Mode: "nli"}
			cfg.HallucinationMitigation.NLIModel.ModelID = "/not-installed/unused-cache-nli"
			cfg.Decisions = []config.Decision{{Name: "route"}}
			if mode == "exact" {
				cfg.Decisions[0].Plugins = []config.DecisionPlugin{{Type: "response_cache", Configuration: config.MustStructuredPayload(config.ResponseCachePluginConfig{Enabled: true, Mode: "exact"})}}
			}
			components, err := buildRouterComponents(cfg)
			require.NoError(t, err)
			router := components.buildRouter()
			t.Cleanup(func() { require.NoError(t, router.Close()) })
			require.False(t, components.embeddings.Ready())
			require.Empty(t, components.modelRuntime.PreparedBindings())
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
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "bert", TargetDimension: 2}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Provider: "http", ExternalModel: "global"}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "openai_compatible", Contract: "embedding.v1"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "global", ModelName: "global", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: endpoint.URL}}}
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
	bindings := components.modelRuntime.PreparedBindings()
	require.Len(t, bindings, 1)
	require.Equal(t, "api.embedding", bindings[0].Identity.Name)
}
