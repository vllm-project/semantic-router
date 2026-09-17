package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

func TestStickyRuntimeRetrievalFingerprintInvalidatesHistoricalSelection(t *testing.T) {
	router, store, ctx := newStickyRuntimeContractRouter(t)
	selection := stickyRuntimeContractSelection("add")
	catalog := stickyRuntimeContractTools("search", "calculate")
	request := &llmprotocol.Request{ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto}}

	first, committed := router.applyStickyToolSelectionWithStatusAndRetrieval(
		request,
		catalog,
		[]llmprotocol.Tool{catalog[0]},
		selection,
		nil,
		"default",
		ctx,
		"retrieval-v1",
	)
	if !committed || len(first) != 1 || first[0].Name != "search" {
		t.Fatalf("first selection = %#v, committed=%v", first, committed)
	}

	ctx.TurnIndex = 1
	second, committed := router.applyStickyToolSelectionWithStatusAndRetrieval(
		request,
		catalog,
		[]llmprotocol.Tool{catalog[1]},
		selection,
		nil,
		"default",
		ctx,
		"retrieval-v2",
	)
	if !committed || len(second) != 1 || second[0].Name != "calculate" {
		t.Fatalf("changed retrieval selection = %#v, committed=%v", second, committed)
	}

	identity := stickyRuntimeContractIdentity(ctx, selection, nil)
	loaded, err := store.Load(context.Background(), identity.StorageKey)
	if err != nil {
		t.Fatalf("load sticky state: %v", err)
	}
	if !loaded.Found || len(loaded.State.Tools) != 1 || loaded.State.Tools[0].Name != "calculate" {
		t.Fatalf("state after retrieval invalidation = %#v", loaded.State)
	}
	if loaded.State.CatalogFingerprint == tools.ToolCatalogFingerprint(catalog) {
		t.Fatal("effective catalog fingerprint should include retrieval metadata")
	}
}

func TestToolSelectionFilterStickyFingerprintTracksEmbeddingConfig(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*config.EmbeddingModels)
	}{
		{
			name: "embedding backend",
			mutate: func(models *config.EmbeddingModels) {
				models.EmbeddingConfig.Backend = config.EmbeddingBackendCandle
			},
		},
		{
			name: "embedding model type",
			mutate: func(models *config.EmbeddingModels) {
				models.EmbeddingConfig.ModelType = config.EmbeddingModelTypeQwen3
			},
		},
		{
			name: "provider endpoint",
			mutate: func(models *config.EmbeddingModels) {
				models.Endpoint.BaseURL = "https://embedding-b.example/v1"
			},
		},
		{
			name: "provider model",
			mutate: func(models *config.EmbeddingModels) {
				models.Endpoint.Model = "embedding-model-b"
			},
		},
		{
			name: "target dimension",
			mutate: func(models *config.EmbeddingModels) {
				models.EmbeddingConfig.TargetDimension = 4
			},
		},
		{
			name: "provider dimension",
			mutate: func(models *config.EmbeddingModels) {
				models.Endpoint.Dimensions = 4
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router, store, ctx := newStickyRuntimeContractRouter(t)
			router.Config.EmbeddingModels = config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					Backend:         config.EmbeddingBackendOpenAICompatible,
					ModelType:       config.EmbeddingModelTypeRemote,
					TargetDimension: 3,
				},
				Endpoint: config.EmbeddingEndpointConfig{
					BaseURL:    "https://embedding-a.example/v1",
					Model:      "embedding-model-a",
					Dimensions: 3,
				},
			}
			router.toolEmbedder = newCachedToolEmbedder(
				&stubToolSelectionEmbeddingProvider{embeddings: map[string][]float32{
					"search query":    {1, 0, 0},
					"search":          {1, 0, 0},
					"calculate query": {0, 1, 0},
					"calculate":       {0, 1, 0},
				}},
				config.EmbeddingModelTypeRemote,
				3,
				"embedding-a.example\x00embedding-model-a",
			)
			selection := stickyRuntimeContractSelection(config.ToolSelectionModeFilter)
			maxNewToolsPerTurn := 0
			selection.Sticky.MaxNewToolsPerTurn = &maxNewToolsPerTurn
			catalog := stickyRuntimeContractTools("search", "calculate")

			before, first := runStickyFilterFingerprintTurn(
				t,
				router,
				store,
				ctx,
				selection,
				catalog,
				"search query",
			)
			if len(first) != 1 || first[0].Name != "search" {
				t.Fatalf("first filter selection = %#v, want search", first)
			}
			test.mutate(&router.Config.EmbeddingModels)
			ctx.TurnIndex = 1
			after, second := runStickyFilterFingerprintTurn(
				t,
				router,
				store,
				ctx,
				selection,
				catalog,
				"calculate query",
			)

			if before == after {
				t.Fatalf("%s change did not invalidate the filter retrieval fingerprint", test.name)
			}
			if len(second) != 1 || second[0].Name != "calculate" {
				t.Fatalf("selection after %s change = %#v, want fresh calculate selection", test.name, second)
			}
			want := tools.EffectiveToolCatalogFingerprint(
				catalog,
				toolsEmbeddingProviderIdentity(router.Config),
			)
			if after != want {
				t.Fatalf("filter catalog fingerprint = %q, want %q", after, want)
			}
		})
	}
}

func runStickyFilterFingerprintTurn(
	t *testing.T,
	router *OpenAIRouter,
	store sessiontools.Store,
	ctx *RequestContext,
	selection *config.ToolSelectionPluginConfig,
	catalog []llmprotocol.Tool,
	classificationText string,
) (string, []llmprotocol.Tool) {
	t.Helper()
	request := &llmprotocol.Request{
		Tools:      append([]llmprotocol.Tool(nil), catalog...),
		ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto},
	}
	if err := router.runToolSelectionPluginFilter(request, classificationText, nil, ctx, selection); err != nil {
		t.Fatalf("runToolSelectionPluginFilter: %v", err)
	}

	identity := stickyRuntimeContractIdentity(ctx, selection, nil)
	loaded, err := store.Load(context.Background(), identity.StorageKey)
	if err != nil {
		t.Fatalf("load sticky state: %v", err)
	}
	if !loaded.Found {
		t.Fatal("filter turn did not commit sticky state")
	}
	return loaded.State.CatalogFingerprint, append([]llmprotocol.Tool(nil), request.Tools...)
}
