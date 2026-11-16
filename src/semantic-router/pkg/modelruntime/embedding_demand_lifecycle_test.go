package modelruntime

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func TestNamedSharedConsumersPrepareTheirGlobalEmbedding(t *testing.T) {
	for _, consumer := range []string{"memory", "vector store"} {
		t.Run(consumer, func(t *testing.T) {
			fake := runtimetest.New(runtimetest.Model{ID: "global", Embedding: &runtimetest.Embedder{Dimensions: []int{8}, Layers: []int{6}}})
			server := httptest.NewServer(fake.Handler())
			defer server.Close()
			cfg := &config.RouterConfig{}
			cfg.EmbeddingDeployment = "global"
			cfg.Memory.Milvus.Dimension = 8
			cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL}}
			if consumer == "vector store" {
				cfg.VectorStore = &config.VectorStoreConfig{Enabled: true, BackendType: "memory"}
			} else {
				cfg.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "active", Profile: config.RoutingProfile{
					Decisions: []config.Decision{{Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(map[string]any{
						"enabled": true, "sources": []any{map[string]any{"type": config.ContextSourceMemory}},
					})}}}},
				}}}
				cfg.Entrypoints = []config.EntrypointMapping{{Recipe: "active", ModelNames: []string{"public"}}}
			}
			manager := modelservice.NewManager()
			defer func() { _ = manager.Shutdown(context.Background()) }()
			lease, err := manager.Acquire(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer lease.Close()
			if len(lease.Statuses()) != 1 {
				t.Fatal("required global model missing from generation plan")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			runtime := serving.New(lease, nil)
			prepared, err := PrepareOwnedGlobalServiceEmbeddings(ctx, cfg, runtime)
			if err != nil {
				t.Fatal(err)
			}
			defer prepared.Close()
			provider, err := prepared.Default()
			if err != nil {
				t.Fatal(err)
			}
			if vector, err := provider.Embed(ctx, consumer+" query"); err != nil || len(vector) != 8 {
				t.Fatalf("required global embedding = %v, %v", vector, err)
			}
			entries := runtime.PreparedBindings()
			if len(entries) != 1 || entries[0].Identity.Recipe != string(config.GlobalModelScope) || entries[0].Identity.Deployment != "global" {
				t.Fatalf("shared consumer used wrong owner: %+v", entries)
			}
		})
	}
}
