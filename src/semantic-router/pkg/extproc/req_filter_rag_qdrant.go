package extproc

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) qdrantCacheConn(reuse bool) *cache.QdrantCache {
	if !reuse || r.Cache == nil {
		return nil
	}
	qc, _ := r.Cache.(*cache.QdrantCache)
	return qc
}

func (r *OpenAIRouter) retrieveFromQdrant(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	qdrantConfig, err := ragConfig.QdrantBackendConfig()
	if err != nil {
		return "", fmt.Errorf("invalid Qdrant RAG config: %w", err)
	}

	qdrantCache := r.qdrantCacheConn(qdrantConfig.ReuseCacheConnection)
	if qdrantCache == nil {
		return "", fmt.Errorf("qdrant connection not available (reuse_cache_connection=%v)", qdrantConfig.ReuseCacheConnection)
	}

	query := ctx.UserContent
	if query == "" {
		return "", fmt.Errorf("user content is empty")
	}

	collectionName := qdrantConfig.Collection
	if collectionName == "" {
		return "", fmt.Errorf("qdrant collection name is required")
	}

	contentField := qdrantConfig.ContentField
	if contentField == "" {
		contentField = "content"
	}

	threshold := float32(0.7)
	if ragConfig.SimilarityThreshold != nil {
		threshold = *ragConfig.SimilarityThreshold
	}

	topK := 5
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}

	queryEmbeddings, err := r.ragQueryEmbeddings(traceCtx, query, ctx)
	if err != nil {
		logging.Errorf("Failed to generate embedding for Qdrant RAG query: %v", err)
		return "", fmt.Errorf("failed to generate embedding")
	}

	var hits ragHits
	for _, queryEmbedding := range queryEmbeddings {
		windowParts, windowScores, searchErr := qdrantCache.SearchCollection(
			traceCtx,
			collectionName,
			queryEmbedding,
			threshold,
			topK,
			contentField,
		)
		if searchErr != nil {
			return "", fmt.Errorf("qdrant search failed: %w", searchErr)
		}
		hits.add(windowParts, windowScores)
	}
	contextParts, scores := hits.top(topK)

	if len(contextParts) == 0 {
		return "", fmt.Errorf("no results above similarity threshold %.3f", threshold)
	}

	bestScore := float32(0.0)
	if len(scores) > 0 {
		bestScore = scores[0]
		ctx.RAGSimilarityScore = bestScore
	}

	logging.Infof("Retrieved %d documents from Qdrant (similarity: %.3f, collection: %s)",
		len(contextParts), bestScore, collectionName)

	return strings.Join(contextParts, "\n\n---\n\n"), nil
}
