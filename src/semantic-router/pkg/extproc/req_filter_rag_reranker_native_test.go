//go:build !windows && cgo && (amd64 || arm64)

package extproc

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// The tiny graph sums actual paired token IDs. This proves the complete
// retrieval -> owned native pair task -> reordered context path, not quality.
func TestRAGStructuredRetrievalUsesActualORTPairScorer(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("actual CPU ORT runtime is required")
	}
	path, err := filepath.Abs("../../../../onnx-binding/instance/testdata/pair_scorer")
	if err != nil {
		t.Fatal(err)
	}
	spec := config.ResolvedModelBinding{Recipe: "a", Name: config.RAGRerankerConsumer, Binding: config.ModelBinding{Deployment: "rank", Contract: config.RelevanceScoresContract, Adapter: "vela_reranker", Head: "model.onnx"}, Deployment: config.ModelDeployment{Provider: "ort", Device: "cpu", Precision: "native", Artifact: path, Input: config.ModelInputBudget{MaxTokens: 32, Overflow: "reject"}}}
	scorer, err := native.New(nil).Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer scorer.Close()
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	manager := vectorstore.NewManager(backend, vectorstore.NewMemoryMetadataRegistry(), 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("fixture"))
	store, err := manager.CreateStore(context.Background(), vectorstore.CreateStoreRequest{Name: "native ranking"})
	if err != nil {
		t.Fatal(err)
	}
	chunks := []vectorstore.EmbeddedChunk{{ID: "first", FileID: "one", Content: "hello", Embedding: []float32{1, 0, 0}}, {ID: "second", FileID: "two", Content: "world", Embedding: []float32{.8, .6, 0}}, {ID: "third", FileID: "three", Content: "秘密", Embedding: []float32{.6, .8, 0}}}
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
	if err != nil || text != "秘密\n\n---\n\nworld" || len(request.RAGRerankScores) != 2 || request.RAGRerankScores[0] != 12 || request.RAGRerankScores[1] != 11 {
		t.Fatalf("native relevance scores did not reorder retrieval: %q %+v %v", text, request.RAGRerankScores, err)
	}
	if request.RAGRerankerIdentity == "" || request.RAGRerankLatency <= 0 {
		t.Fatal("actual inference observation missing")
	}
}
