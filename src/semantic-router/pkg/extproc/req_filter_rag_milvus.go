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

// retrieveFromMilvus retrieves context from Milvus backend
func (r *OpenAIRouter) retrieveFromMilvus(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	milvusConfig, ok := ragConfig.BackendConfig.(*config.MilvusRAGConfig)
	if !ok {
		return "", fmt.Errorf("invalid Milvus RAG config")
	}

	// Get Milvus cache instance
	var milvusCache *cache.MilvusCache
	if milvusConfig.ReuseCacheConnection && r.Cache != nil {
		// Try to reuse existing connection
		if mc, ok := r.Cache.(*cache.MilvusCache); ok {
			milvusCache = mc
		}
	}

	if milvusCache == nil {
		return "", fmt.Errorf("Milvus connection not available (reuse_cache_connection=%v)", milvusConfig.ReuseCacheConnection)
	}

	// Perform similarity search
	query := ctx.UserContent
	if query == "" {
		return "", fmt.Errorf("user content is empty")
	}

	threshold := float32(0.7) // Default
	if ragConfig.SimilarityThreshold != nil {
		threshold = *ragConfig.SimilarityThreshold
	}

	topK := 5 // Default
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}

	// Generate embedding for query
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		return "", fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Build filter expression
	filterExpr := milvusConfig.FilterExpression
	if filterExpr == "" {
		filterExpr = "response_body != \"\"" // Only get complete entries
	}

	// Get collection name
	collectionName := milvusConfig.Collection
	if collectionName == "" {
		return "", fmt.Errorf("Milvus collection name is required")
	}

	// Determine content field
	contentField := milvusConfig.ContentField
	if contentField == "" {
		contentField = "content"
	}

	// Use MilvusCache SearchDocuments method
	contextParts, scores, err := milvusCache.SearchDocuments(
		traceCtx,
		collectionName,
		queryEmbedding,
		threshold,
		topK,
		filterExpr,
		contentField,
	)

	if err != nil {
		return "", fmt.Errorf("Milvus search failed: %w", err)
	}

	if len(contextParts) == 0 {
		return "", fmt.Errorf("no results above similarity threshold %.3f", threshold)
	}

	// Combine context parts
	retrievedContext := strings.Join(contextParts, "\n\n---\n\n")

	// Store best similarity score
	bestScore := float32(0.0)
	if len(scores) > 0 {
		bestScore = scores[0] // Scores are already sorted
		ctx.RAGSimilarityScore = bestScore
	}

	logging.Infof("Retrieved %d documents from Milvus (similarity: %.3f, collection: %s)",
		len(contextParts), bestScore, collectionName)

	return retrievedContext, nil
}
