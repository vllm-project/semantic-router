package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
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

func TestRAGStructuredRetrievalUsesDecisionScoresAndStableTies(t *testing.T) {
	runtime, fake := decisionRAGRuntime(t)
	spec := config.ResolvedModelBinding{Recipe: "a", Name: config.RAGRerankerConsumer, Binding: config.ModelBinding{Deployment: "rank", Contract: config.DecisionTaskContract}, Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://runtime:8100", Input: config.ModelInputBudget{Overflow: "reject"}}}
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
	if err != nil || text != "秘密\n\n---\n\nhello there" || len(request.RAGRerankScores) != 2 || request.RAGRerankScores[0] != 3 || request.RAGRerankScores[1] != 3 {
		t.Fatalf("runtime relevance scores did not reorder retrieval: %q %+v %v", text, request.RAGRerankScores, err)
	}
	if request.RAGRerankerIdentity == "" || request.RAGRerankLatency <= 0 {
		t.Fatal("actual inference observation missing")
	}
	if request.RAGRerankScoreType != "decision_relevance" || fake.Calls("rerank") != 0 {
		t.Fatal("decision scorer mislabeled or called native rerank")
	}
	calls, _ := fake.Bundles()
	if calls == 0 {
		t.Fatal("candidate questions did not use runtime bundling")
	}
}

// Override fake scores to prove ranking and ties through real transport. This
// fixture provides no model-quality evidence.
func decisionRAGRuntime(t *testing.T) (*serving.Runtime, *runtimetest.Runtime) {
	t.Helper()
	fake := runtimetest.New(runtimetest.Model{ID: "rank", ModelSHA256: strings.Repeat("a", 64)})
	handler := fake.Handler()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, req)
		var payload any
		if json.Unmarshal(recorder.Body.Bytes(), &payload) != nil {
			t.Error("invalid fixture JSON")
			return
		}
		var visit func(any)
		visit = func(value any) {
			switch value := value.(type) {
			case map[string]any:
				if answers, ok := value["answers"].(map[string]any); ok {
					for id, raw := range answers {
						if !strings.HasPrefix(id, "rag.reranker:candidate:") {
							continue
						}
						index, err := strconv.Atoi(strings.TrimPrefix(id, "rag.reranker:candidate:"))
						if err != nil {
							t.Error(err)
							continue
						}
						answer := raw.(map[string]any)
						probabilities := map[string]float64{"0": 0, "1": 0, "2": 0, "3": 0}
						if index == 0 {
							probabilities["1"] = 1
						} else {
							probabilities["3"] = 1
						}
						answer["probabilities"], answer["confidence"] = probabilities, 1.0
						if index == 0 {
							answer["score"] = 1.0
						} else {
							answer["score"] = 3.0
						}
					}
				}
				for _, child := range value {
					visit(child)
				}
			case []any:
				for _, child := range value {
					visit(child)
				}
			}
		}
		visit(payload)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(recorder.Code)
		if err := json.NewEncoder(w).Encode(payload); err != nil {
			t.Error(err)
		}
	}))
	t.Cleanup(server.Close)
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"rank": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL}})
	if err != nil {
		t.Fatal(err)
	}
	return serving.New(lease, nil), fake
}
