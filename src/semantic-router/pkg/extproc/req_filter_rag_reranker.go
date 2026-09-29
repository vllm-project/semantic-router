package extproc

import (
	"context"
	"fmt"
	"sort"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// rerankVectorStoreResults scores complete native pairs before formatting or
// context truncation. Similarities and stable file/chunk identities stay intact.
func (r *OpenAIRouter) rerankVectorStoreResults(traceCtx context.Context, ctx *RequestContext, cfg *config.RAGPluginConfig, query string, results []vectorstore.SearchResult) ([]vectorstore.SearchResult, error) {
	if cfg.Rerank == nil || len(results) == 0 {
		return results, nil
	}
	recipe := ctx.Routing.RecipeName()
	scorer := r.rerankers[recipe]
	if scorer == nil {
		return nil, fmt.Errorf("RAG reranker is not prepared for recipe %q", recipe)
	}
	pairs := make([]tasks.QueryDocument, len(results))
	for i, result := range results {
		pairs[i] = tasks.QueryDocument{Query: query, Document: result.Content}
	}
	start := time.Now()
	output, err := scorer.ScorePairs(traceCtx, string(recipe), pairs)
	ctx.RAGRerankLatency = time.Since(start)
	span := trace.SpanFromContext(traceCtx)
	span.SetAttributes(attribute.Float64("rag.rerank_latency_seconds", ctx.RAGRerankLatency.Seconds()), attribute.Int("rag.rerank_candidates", len(pairs)))
	if err != nil {
		return nil, fmt.Errorf("RAG pair scoring failed: %w", err)
	}
	if err = tasks.ValidateRelevanceScores(pairs, output); err != nil {
		return nil, fmt.Errorf("RAG pair scoring returned invalid output: %w", err)
	}
	if err = traceCtx.Err(); err != nil {
		return nil, err
	}
	indices := make([]int, len(results))
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool { return output.Scores[indices[i]] > output.Scores[indices[j]] })
	if cfg.Rerank.TopK != nil && len(indices) > *cfg.Rerank.TopK {
		indices = indices[:*cfg.Rerank.TopK]
	}
	reordered := make([]vectorstore.SearchResult, len(indices))
	ctx.RAGRerankScores = make([]float32, len(indices))
	for i, index := range indices {
		reordered[i] = results[index]
		ctx.RAGRerankScores[i] = output.Scores[index]
	}
	ctx.RAGRerankerIdentity = scorer.CacheIdentity()
	span.SetAttributes(attribute.String("rag.reranker_identity", ctx.RAGRerankerIdentity), attribute.String("rag.rerank_score_type", "relevance_logit"), attribute.Float64Slice("rag.rerank_scores", float64Relevance(ctx.RAGRerankScores)))
	return reordered, nil
}

func float64Relevance(scores []float32) []float64 {
	values := make([]float64, len(scores))
	for i, score := range scores {
		values[i] = float64(score)
	}
	return values
}
