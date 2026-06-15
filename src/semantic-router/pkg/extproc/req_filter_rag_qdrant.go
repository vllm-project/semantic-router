package extproc

import (
	"context"
	"fmt"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
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

	queryEmbedding, err := candle_binding.GetEmbedding(query, 0)
	if err != nil {
		logging.Errorf("Failed to generate embedding for Qdrant RAG query: %v", err)
		return "", fmt.Errorf("failed to generate embedding")
	}

	contextParts, scores, err := qdrantCache.SearchCollection(
		traceCtx,
		collectionName,
		queryEmbedding,
		threshold,
		topK,
		contentField,
	)
	if err != nil {
		return "", fmt.Errorf("qdrant search failed: %w", err)
	}

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
