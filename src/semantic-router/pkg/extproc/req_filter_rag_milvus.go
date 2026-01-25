package extproc

import (
	"context"
	"fmt"
	"regexp"
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
		return "", fmt.Errorf("milvus connection not available (reuse_cache_connection=%v)", milvusConfig.ReuseCacheConnection)
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
		// Log full error internally but don't expose it to avoid information disclosure
		logging.Errorf("Failed to generate embedding for RAG query: %v", err)
		return "", fmt.Errorf("failed to generate embedding")
	}

	// Get collection name
	collectionName := milvusConfig.Collection
	if collectionName == "" {
		return "", fmt.Errorf("milvus collection name is required")
	}

	// Determine content field
	contentField := milvusConfig.ContentField
	if contentField == "" {
		contentField = "content"
	}

	// Build filter expression
	filterExpr := milvusConfig.FilterExpression
	if filterExpr == "" {
		// Use contentField for default filter if available
		if contentField != "" {
			filterExpr = fmt.Sprintf("%s != \"\"", contentField)
		}
		// If no contentField, leave filterExpr empty (no filtering)
	} else {
		// Validate filter expression to prevent injection attacks
		if validateErr := validateMilvusFilterExpression(filterExpr); validateErr != nil {
			return "", fmt.Errorf("invalid filter expression: %w", validateErr)
		}
	}

	// Use MilvusCache SearchDocuments method
	// Pass empty strings/0 for vector field config to use cache defaults
	// If RAG collection has different config, these can be added to MilvusRAGConfig
	contextParts, scores, err := milvusCache.SearchDocuments(
		traceCtx,
		collectionName,
		queryEmbedding,
		threshold,
		topK,
		filterExpr,
		contentField,
		"", // vectorFieldName - use cache default
		"", // metricType - use cache default
		0,  // ef - use cache default
	)
	if err != nil {
		return "", fmt.Errorf("milvus search failed: %w", err)
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

// validateMilvusFilterExpression validates a Milvus filter expression to prevent injection attacks
// This performs basic validation - for production use, consider using Milvus's parameterized query API
// or a more sophisticated expression parser.
func validateMilvusFilterExpression(filterExpr string) error {
	if filterExpr == "" {
		return nil
	}

	// Check for potentially dangerous patterns
	dangerousPatterns := []string{
		";",        // SQL injection attempts
		"--",       // SQL comments
		"/*",       // SQL block comments
		"*/",       // SQL block comments
		"DROP",     // SQL DROP commands
		"DELETE",   // SQL DELETE commands
		"UPDATE",   // SQL UPDATE commands
		"INSERT",   // SQL INSERT commands
		"EXEC",     // SQL EXEC commands
		"EXECUTE",  // SQL EXECUTE commands
		"CREATE",   // SQL CREATE commands
		"ALTER",    // SQL ALTER commands
		"TRUNCATE", // SQL TRUNCATE commands
		"GRANT",    // SQL GRANT commands
		"REVOKE",   // SQL REVOKE commands
		"UNION",    // SQL UNION attacks
		"SELECT",   // SQL SELECT (though Milvus uses SELECT, we want to be careful)
		"\\x",      // Hex encoding attempts
		"\\u",      // Unicode encoding attempts
	}

	upperExpr := strings.ToUpper(filterExpr)
	for _, pattern := range dangerousPatterns {
		if strings.Contains(upperExpr, pattern) {
			return fmt.Errorf("filter expression contains potentially dangerous pattern: %s", pattern)
		}
	}

	// Validate basic Milvus expression syntax
	// Milvus expressions typically follow: field operator value [logical_operator field operator value]
	// Valid operators: ==, !=, >, >=, <, <=, in, not in, like, not like
	// Valid logical operators: &&, ||, and, or, not
	validOperatorPattern := regexp.MustCompile(`\s*(==|!=|>|>=|<|<=|in|not\s+in|like|not\s+like|&&|\|\||and|or|not)\s*`)

	// Check for balanced parentheses
	parenCount := 0
	for _, char := range filterExpr {
		switch char {
		case '(':
			parenCount++
		case ')':
			parenCount--
			if parenCount < 0 {
				return fmt.Errorf("unbalanced parentheses in filter expression")
			}
		}
	}
	if parenCount != 0 {
		return fmt.Errorf("unbalanced parentheses in filter expression")
	}

	// Check for reasonable length (prevent DoS via extremely long expressions)
	const maxFilterLength = 10000
	if len(filterExpr) > maxFilterLength {
		return fmt.Errorf("filter expression exceeds maximum length of %d characters", maxFilterLength)
	}

	// Basic syntax check: should contain at least one comparison operator
	hasOperator := validOperatorPattern.MatchString(filterExpr)
	if !hasOperator && len(strings.Fields(filterExpr)) > 0 {
		// If it has content but no operators, it might be malformed
		// Allow simple field checks like "field_name" (Milvus supports this)
		if !regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`).MatchString(strings.TrimSpace(filterExpr)) {
			return fmt.Errorf("filter expression appears to be malformed")
		}
	}

	return nil
}
