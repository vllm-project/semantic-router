package modelruntime

import (
	"context"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func TestEmbeddingDemandAvoidsStartingUnusedManagedModels(t *testing.T) {
	for _, consumer := range []string{"unused KB catalog", "unused ML artifacts", "KB signal", "KB projection"} {
		t.Run(consumer, func(t *testing.T) {
			// If demand planning tries to start the unused global model, Acquire
			// fails instead of downloading a real model or starting a subprocess.
			t.Setenv(modelservice.RuntimeCommandEnv, filepath.Join(t.TempDir(), "must-not-start"))
			t.Setenv(modelservice.RuntimeDirEnv, t.TempDir())
			fake := runtimetest.New(runtimetest.Model{ID: "local", Embedding: &runtimetest.Embedder{Dimensions: []int{8}, Layers: []int{6}}})
			server := httptest.NewServer(fake.Handler())
			defer server.Close()
			cfg := &config.RouterConfig{KnowledgeBases: []config.KnowledgeBaseConfig{{Name: "catalog-kb"}}}
			cfg.EmbeddingConfig.ModelType = "mmbert"
			cfg.MmBertModelPath = "/does-not-exist/unused-embedding"
			cfg.ModelDeployments = map[string]config.ModelDeployment{
				"global": {Provider: config.ModelRuntimeProvider, Artifact: "test/must-not-download", Device: "cpu"},
				"local":  {Provider: config.ModelRuntimeProvider, Endpoint: server.URL},
			}
			cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Contract: "embedding.v1"}}
			cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "local", Contract: "embedding.v1"}}
			active := false
			switch consumer {
			case "unused ML artifacts":
				cfg.ModelSelection.Enabled = true
				cfg.ModelSelection.ML.ModelsPath = "unused-selectors"
			case "KB signal":
				cfg.KBRules = []config.KBSignalRule{{KB: "catalog-kb"}}
				active = true
			case "KB projection":
				cfg.Projections.Scores = []config.ProjectionScore{{Name: "quality", Inputs: []config.ProjectionScoreInput{{Type: config.ProjectionInputKBMetric, KB: "catalog-kb"}}}}
				active = true
			}
			manager := modelservice.NewManager()
			defer func() { _ = manager.Shutdown(context.Background()) }()
			lease, err := manager.Acquire(cfg)
			if err != nil {
				t.Fatalf("unused managed model was started: %v", err)
			}
			defer lease.Close()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if waitErr := lease.WaitManaged(ctx); waitErr != nil {
				t.Fatal(waitErr)
			}
			runtime := serving.New(lease, nil)
			prepared, err := PrepareOwnedEmbeddings(ctx, cfg, runtime)
			if err != nil {
				t.Fatal(err)
			}
			defer prepared.Close()
			if prepared.Ready() != active {
				t.Fatalf("embedding ready = %v, want %v", prepared.Ready(), active)
			}
			if !active {
				if len(lease.Statuses()) != 0 || len(runtime.PreparedBindings()) != 0 || fake.Calls("embeddings") != 0 {
					t.Fatal("unused catalog caused model initialization or embedding requests")
				}
				return
			}
			provider, err := prepared.Default()
			if err != nil {
				t.Fatal(err)
			}
			vector, err := provider.Embed(ctx, "knowledge base query")
			if err != nil || len(vector) != 8 {
				t.Fatalf("active KB embedding = %v, %v", vector, err)
			}
			entries := runtime.PreparedBindings()
			if len(entries) != 1 || entries[0].Identity.Deployment != "local" {
				t.Fatalf("active KB borrowed an unused global model: %+v", entries)
			}
		})
	}
}

func TestOwnedMLPreparesSelectedEmbeddingWithoutUnusedPrimary(t *testing.T) {
	services := newEmbeddingRuntime("@embedding.qwen3")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = "/does-not-exist/unused-primary"
	cfg.Qwen3ModelPath = t.TempDir()
	cfg.ModelSelection.Enabled = true
	cfg.ModelSelection.ML.ModelsPath, cfg.ModelSelection.ML.ModelType = "selectors", "qwen3"
	cfg.Decisions = []config.Decision{{Algorithm: &config.AlgorithmConfig{Type: "knn"}}}
	runtime := serving.New(services, nil)
	prepared, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	provider, err := prepared.Get("qwen3", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if vector, err := provider.Embed(context.Background(), "routing query"); err != nil || len(vector) != 8 {
		t.Fatalf("selected ML embedding = %v, %v", vector, err)
	}
	if _, err := prepared.Get("mmbert", 0, 0); err == nil {
		t.Fatal("unused primary was prepared")
	}
	if entries := runtime.PreparedBindings(); len(entries) != 1 || entries[0].Identity.Deployment != "@embedding.qwen3" {
		t.Fatalf("prepared bindings = %+v", entries)
	}
}

func TestNamedSharedConsumersPrepareTheirGlobalEmbedding(t *testing.T) {
	for _, consumer := range []string{"tool_selection", "memory", "embedding API"} {
		t.Run(consumer, func(t *testing.T) {
			fake := runtimetest.New(runtimetest.Model{ID: "global", Embedding: &runtimetest.Embedder{Dimensions: []int{8}, Layers: []int{6}}})
			server := httptest.NewServer(fake.Handler())
			defer server.Close()
			cfg := &config.RouterConfig{RouterOptions: config.RouterOptions{AutoModelNames: []string{}}}
			cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "mmbert", TargetDimension: 8, TargetLayer: 6}
			cfg.Memory.Milvus.Dimension = 8
			cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL}}
			cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Contract: "embedding.v1"}}
			if consumer == "embedding API" {
				cfg.API.Embeddings.Enabled = true
			} else {
				cfg.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "active", Profile: config.RoutingProfile{
					Decisions: []config.Decision{{Plugins: []config.DecisionPlugin{{Type: consumer, Configuration: config.MustStructuredPayload(map[string]any{"enabled": true})}}}},
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
