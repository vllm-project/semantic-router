package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// The fake reranker scores query-word overlap. This proves the complete
// retrieval -> model_runtime pair task -> reordered context path, not quality.
func TestRAGStructuredRetrievalReordersByRuntimeRelevance(t *testing.T) {
	runtime, _ := servingtest.Runtime(t, map[string]runtimetest.Model{
		"rank": {Rerank: &runtimetest.Reranker{Default: api.RerankExit{Layer: 22, Dimension: 768}}},
	})
	spec := config.ResolvedModelBinding{Recipe: "a", Name: config.RAGRerankerConsumer, Binding: config.ModelBinding{Deployment: "rank", Contract: config.RelevanceScoresContract}, Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://runtime:8100", Input: config.ModelInputBudget{Overflow: "reject"}}}
	scorer, err := runtime.Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer scorer.Close()
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	manager := vectorstore.NewManager(backend, vectorstore.NewMemoryMetadataRegistry(), 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("fixture"))
	store, err := manager.CreateStore(context.Background(), vectorstore.CreateStoreRequest{Name: "runtime ranking"})
	if err != nil {
		t.Fatal(err)
	}
	chunks := []vectorstore.EmbeddedChunk{{ID: "first", FileID: "one", Content: "world", Embedding: []float32{1, 0, 0}}, {ID: "second", FileID: "two", Content: "秘密", Embedding: []float32{.8, .6, 0}}, {ID: "third", FileID: "three", Content: "hello there", Embedding: []float32{.6, .8, 0}}}
	if err = backend.InsertChunks(context.Background(), store.ID, chunks); err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{Manager: manager, Embedder: ragRetrievalEmbedder{}})
	r := &OpenAIRouter{RuntimeRegistry: registry, rerankers: map[config.RecipeName]modelruntime.PairScorer{"a": scorer}}
	two, three, zero := 2, 3, float32(0)
	cfg := &config.RAGPluginConfig{Enabled: true, Backend: "vectorstore", TopK: &three, SimilarityThreshold: &zero, Rerank: &config.RAGRerankConfig{TopK: &two}, BackendConfig: config.MustStructuredPayload(config.VectorStoreRAGConfig{VectorStoreID: store.ID})}
	request := &RequestContext{UserContent: "hello"}
	request.Routing.SelectRecipe(&config.RoutingRecipe{Name: "a"})
	text, err := r.retrieveContext(context.Background(), request, cfg)
	if err != nil || text != "hello there\n\n---\n\nworld" || len(request.RAGRerankScores) != 2 || request.RAGRerankScores[0] != 2 || request.RAGRerankScores[1] != -2 {
		t.Fatalf("runtime relevance scores did not reorder retrieval: %q %+v %v", text, request.RAGRerankScores, err)
	}
	if request.RAGRerankerIdentity == "" || request.RAGRerankLatency <= 0 {
		t.Fatal("actual inference observation missing")
	}
}
