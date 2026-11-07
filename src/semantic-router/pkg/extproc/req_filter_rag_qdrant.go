package extproc

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) qdrantCacheConn(reuse bool) *cache.QdrantCache {
	if !reuse || r.Cache == nil {
		return nil
	}
	qc, _ := r.Cache.(*cache.QdrantCache)
	return qc
}

// recallFromQdrant recalls documents from a Qdrant collection over the
// response cache's connection.
func (r *OpenAIRouter) recallFromQdrant(traceCtx context.Context, ctx *RequestContext, source *config.ContextSource, query string, recall config.MatchRecall) ([]match.Scored, error) {
	qdrantConfig, err := source.QdrantBackendConfig()
	if err != nil {
		return nil, fmt.Errorf("invalid Qdrant RAG config: %w", err)
	}
	qdrantCache := r.qdrantCacheConn(qdrantConfig.ReuseCacheConnection)
	if qdrantCache == nil {
		return nil, fmt.Errorf("qdrant connection not available (reuse_cache_connection=%v)", qdrantConfig.ReuseCacheConnection)
	}
	if qdrantConfig.Collection == "" {
		return nil, fmt.Errorf("qdrant collection name is required")
	}
	contentField := qdrantConfig.ContentField
	if contentField == "" {
		contentField = "content"
	}
	queryEmbedding, err := r.ragQueryEmbedding(traceCtx, query, ctx)
	if err != nil {
		logging.Errorf("Failed to generate embedding for Qdrant RAG query: %v", err)
		return nil, fmt.Errorf("failed to generate embedding")
	}
	contextParts, scores, err := qdrantCache.SearchCollection(
		traceCtx,
		qdrantConfig.Collection,
		queryEmbedding,
		recallThreshold(recall),
		recall.TopK,
		contentField,
	)
	if err != nil {
		return nil, fmt.Errorf("qdrant search failed: %w", err)
	}
	logging.Infof("Recalled %d documents from Qdrant (collection: %s)", len(contextParts), qdrantConfig.Collection)
	return scoredDocuments(contextParts, float64Scores(scores)), nil
}
