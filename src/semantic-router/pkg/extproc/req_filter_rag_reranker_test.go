package extproc

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

type ragTestScorer struct {
	scores   []float32
	identity string
	err      error
	calls    int
	pairs    []tasks.QueryDocument
}

func (s *ragTestScorer) Close() error          { return nil }
func (s *ragTestScorer) CacheIdentity() string { return s.identity }
func (s *ragTestScorer) ScorePairs(ctx context.Context, recipe string, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
	s.calls++
	s.pairs = append([]tasks.QueryDocument(nil), pairs...)
	if err := ctx.Err(); err != nil {
		return tasks.RelevanceScores{}, err
	}
	if s.err != nil {
		return tasks.RelevanceScores{}, s.err
	}
	usage := make([]tasks.InputUsage, len(s.scores))
	for i := range usage {
		usage[i] = tasks.InputUsage{OriginalTokens: 5, ProcessedTokens: 5}
	}
	return tasks.RelevanceScores{Scores: s.scores, Inputs: usage}, nil
}

func TestRAGRelevanceStableOrderIdentitiesAndError(t *testing.T) {
	three := 3
	cfg := &config.RAGPluginConfig{Enabled: true, Backend: "vectorstore", Rerank: &config.RAGRerankConfig{TopK: &three}}
	scorer := &ragTestScorer{scores: []float32{-2, 9, 9, 4}, identity: "weights-head-contract"}
	r := &OpenAIRouter{rerankers: map[config.RecipeName]modelruntime.PairScorer{"a": scorer}}
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "a"})
	hits := []vectorstore.SearchResult{{FileID: "first", ChunkIndex: 0, Content: "A", Score: .99}, {FileID: "shared", ChunkIndex: 1, Content: "B", Score: .8}, {FileID: "shared", ChunkIndex: 2, Content: "C", Score: .7}, {FileID: "last", Content: "D", Score: .6}}
	original := append([]vectorstore.SearchResult(nil), hits...)
	got, err := r.rerankVectorStoreResults(context.Background(), ctx, cfg, "query", hits)
	if err != nil {
		t.Fatal(err)
	}
	want := []vectorstore.SearchResult{hits[1], hits[2], hits[3]}
	if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(hits, original) {
		t.Fatalf("ID/tie/score mutation: %+v", got)
	}
	text, similarity, _ := formatVectorStoreRetrievalResults(got)
	if text != "B\n\n---\n\nC\n\n---\n\nD" || similarity != .8 || ctx.RAGRerankScores[0] != 9 || ctx.RAGRerankerIdentity != scorer.identity {
		t.Fatal("raw relevance replaced retrieval metadata")
	}
	if scorer.pairs[1] != (tasks.QueryDocument{Query: "query", Document: "B"}) {
		t.Fatal("pair template input was concatenated")
	}
	for _, kind := range []string{"arity", "nan", "error", "cancel", "foreign"} {
		t.Run(kind, func(t *testing.T) {
			local := *scorer
			local.scores = append([]float32(nil), scorer.scores...)
			r.rerankers["a"] = &local
			request := &RequestContext{}
			request.Routing.SelectRecipe(&config.RoutingRecipe{Name: "a"})
			traceCtx := context.Background()
			switch kind {
			case "arity":
				local.scores = local.scores[:1]
			case "nan":
				local.scores[0] = float32(math.NaN())
			case "error":
				local.err = errors.New("provider failed")
			case "cancel":
				var cancel context.CancelFunc
				traceCtx, cancel = context.WithCancel(traceCtx)
				cancel()
			case "foreign":
				request.Routing.SelectRecipe(&config.RoutingRecipe{Name: "b"})
			}
			if output, err := r.rerankVectorStoreResults(traceCtx, request, cfg, "query", hits); err == nil || output != nil {
				t.Fatalf("failed ranker produced context: %+v %v", output, err)
			}
		})
	}
}

func TestRAGRerankCacheBindsRecipeAndActualScorer(t *testing.T) {
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	manager := vectorstore.NewManager(backend, vectorstore.NewMemoryMetadataRegistry(), 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("embedding"))
	store, err := manager.CreateStore(context.Background(), vectorstore.CreateStoreRequest{Name: "ranking"})
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{Manager: manager})
	a, b := &ragTestScorer{identity: "weights-A"}, &ragTestScorer{identity: "weights-B"}
	r := &OpenAIRouter{RuntimeRegistry: registry, rerankers: map[config.RecipeName]modelruntime.PairScorer{"a": a, "b": a}}
	cfg := &config.RAGPluginConfig{Backend: "vectorstore", CacheResults: true, Rerank: &config.RAGRerankConfig{}, BackendConfig: config.MustStructuredPayload(config.VectorStoreRAGConfig{VectorStoreID: store.ID})}
	r.setRAGCache("a", "q", "ranked-A", cfg)
	if text, hit := r.getRAGCache("a", "q", cfg); !hit || text != "ranked-A" {
		t.Fatal("current ranking missed")
	}
	if _, hit := r.getRAGCache("b", "q", cfg); hit {
		t.Fatal("recipe cache crossed owner")
	}
	r.rerankers["a"] = b
	if _, hit := r.getRAGCache("a", "q", cfg); hit {
		t.Fatal("old scorer context reused")
	}
	delete(r.rerankers, "a")
	if key := r.buildRAGCacheKey("a", "q", cfg); key != "" {
		t.Fatal("unavailable scorer bypassed by cache")
	}
	r.rerankers["a"] = a
	if _, hit := r.getRAGCache("a", "q", cfg); !hit {
		t.Fatal("original cache was destroyed")
	}
}

type ragRetrievalEmbedder struct{}

func (ragRetrievalEmbedder) Embed(context.Context, string) ([]float32, error) {
	return []float32{1, 0, 0}, nil
}
func (ragRetrievalEmbedder) Dimension() int { return 3 }

func TestRAGStructuredRetrievalInvokesScorerAndCachesOnlyCurrentOrder(t *testing.T) {
	backend := vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})
	manager := vectorstore.NewManager(backend, vectorstore.NewMemoryMetadataRegistry(), 3, vectorstore.BackendTypeMemory, vectorstore.WithEmbeddingIdentity("actual-embedding"))
	store, err := manager.CreateStore(context.Background(), vectorstore.CreateStoreRequest{Name: "documents"})
	if err != nil {
		t.Fatal(err)
	}
	chunks := []vectorstore.EmbeddedChunk{
		{ID: "a", FileID: "shared", ChunkIndex: 0, Content: "A", Embedding: []float32{1, 0, 0}},
		{ID: "b", FileID: "shared", ChunkIndex: 1, Content: "B", Embedding: []float32{.8, .6, 0}},
		{ID: "c", FileID: "third", ChunkIndex: 0, Content: "C", Embedding: []float32{.6, .8, 0}},
	}
	if err = backend.InsertChunks(context.Background(), store.ID, chunks); err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{Manager: manager, Embedder: ragRetrievalEmbedder{}})
	scorer := &ragTestScorer{scores: []float32{-2, 9, 4}, identity: "old-scorer"}
	r := &OpenAIRouter{RuntimeRegistry: registry, rerankers: map[config.RecipeName]modelruntime.PairScorer{"a": scorer}}
	three, two, threshold := 3, 2, float32(0)
	cfg := &config.RAGPluginConfig{Enabled: true, Backend: "vectorstore", TopK: &three, SimilarityThreshold: &threshold, CacheResults: true, Rerank: &config.RAGRerankConfig{TopK: &two}, BackendConfig: config.MustStructuredPayload(config.VectorStoreRAGConfig{VectorStoreID: store.ID})}
	request := &RequestContext{UserContent: "query"}
	request.Routing.SelectRecipe(&config.RoutingRecipe{Name: "a"})
	text, err := r.retrieveContext(context.Background(), request, cfg)
	if err != nil || text != "B\n\n---\n\nC" || scorer.calls != 1 {
		t.Fatalf("real retrieval did not rerank before formatting: %q %v calls=%d", text, err, scorer.calls)
	}
	if _, err = r.retrieveContext(context.Background(), request, cfg); err != nil || scorer.calls != 1 {
		t.Fatal("cache hit reran model", err)
	}
	replacement := &ragTestScorer{scores: []float32{8, 2, 1}, identity: "new-scorer"}
	r.rerankers["a"] = replacement
	text, err = r.retrieveContext(context.Background(), request, cfg)
	if err != nil || text != "A\n\n---\n\nB" || replacement.calls != 1 {
		t.Fatalf("new model reused stale context: %q %v", text, err)
	}
	replacement.identity = "failed-model"
	replacement.err = errors.New("native scorer failed")
	if text, err = r.retrieveContext(context.Background(), request, cfg); err == nil || text != "" {
		t.Fatal("failed scorer injected old retrieval", text, err)
	}
}
