package extproc

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/openai"
)

// retrieveFromOpenAI retrieves context using OpenAI's file_search tool
// This supports two modes:
// 1. Tool-based (Responses API workflow): Adds file_search tool to request, LLM calls it
// 2. Direct search: Uses vector store search API for synchronous retrieval
func (r *OpenAIRouter) retrieveFromOpenAI(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	openaiConfig, err := ragConfig.OpenAIBackendConfig()
	if err != nil {
		return "", fmt.Errorf("invalid OpenAI RAG config: %w", err)
	}

	baseURL := openaiConfig.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}

	query := ctx.UserContent
	if query == "" {
		return "", fmt.Errorf("user content is empty")
	}

	// Determine workflow mode
	workflowMode := openaiConfig.WorkflowMode
	if workflowMode == "" {
		workflowMode = "direct_search" // Default to direct search for synchronous retrieval
	}

	// For tool-based workflow (Responses API), add file_search tool to request
	// The LLM will call it and results will be in response annotations
	if workflowMode == "tool_based" {
		logging.Infof("OpenAI RAG: Using tool-based workflow (Responses API), adding file_search tool")
		if toolErr := r.addFileSearchToolToRequest(ctx, openaiConfig); toolErr != nil {
			return "", fmt.Errorf("failed to add file_search tool: %w", toolErr)
		}
		// Return empty - context will be retrieved from tool response annotations
		// This requires response handling to extract context from annotations
		return "", nil
	}

	// For direct_search workflow, use vector store search API for synchronous retrieval
	// This maintains backward compatibility with existing injection modes
	logging.Infof("OpenAI RAG: Using direct search workflow (vector_store_id: %s)", openaiConfig.VectorStoreID)

	// Create vector store client
	vectorStoreClient := openai.NewVectorStoreClient(baseURL, openaiConfig.APIKey)

	// Determine search parameters
	limit := 20 // Default
	if openaiConfig.MaxNumResults != nil {
		limit = *openaiConfig.MaxNumResults
	}

	// Perform vector store search
	filterMap, err := openaiConfig.FilterMap()
	if err != nil {
		return "", fmt.Errorf("invalid OpenAI filter config: %w", err)
	}
	searchResp, err := vectorStoreClient.SearchVectorStore(traceCtx, openaiConfig.VectorStoreID, query, limit, filterMap)
	if err != nil {
		return "", fmt.Errorf("vector store search failed: %w", err)
	}

	if len(searchResp.Data) == 0 {
		return "", fmt.Errorf("no results found in vector store")
	}

	// Extract content from search results
	var contexts []string
	bestScore := float64(0.0)
	for _, result := range searchResp.Data {
		if result.Content != "" {
			contexts = append(contexts, result.Content)
			if result.Score > bestScore {
				bestScore = result.Score
			}
		}
	}

	if len(contexts) == 0 {
		return "", fmt.Errorf("no content found in search results")
	}

	// Combine contexts
	retrievedContext := strings.Join(contexts, "\n\n---\n\n")

	// Store best similarity score
	ctx.RAGSimilarityScore = float32(bestScore)

	logging.Infof("Retrieved %d documents from OpenAI vector store (similarity: %.3f, vector_store_id: %s)",
		len(contexts), bestScore, openaiConfig.VectorStoreID)

	return retrievedContext, nil
}

// addFileSearchToolToRequest adds the file_search tool to the request
// This follows the Responses API workflow where tools are part of the request
func (r *OpenAIRouter) addFileSearchToolToRequest(ctx *RequestContext, openaiConfig *config.OpenAIRAGConfig) error {
	return llmprotocol.NewError(
		llmprotocol.ErrorUnsupportedFeature,
		"server_hosted_tool_unsupported",
		"server-hosted file search is not available on this backend interface; use direct_search",
		nil,
	)
}

// NOTE: Tool-based workflow functions (handleFileSearchToolCall, extractContextFromFileSearchResults)
// are reserved for future implementation when response annotation extraction is added.
// For now, use "direct_search" workflow_mode for synchronous retrieval.
