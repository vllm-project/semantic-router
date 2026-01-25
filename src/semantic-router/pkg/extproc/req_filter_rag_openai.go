package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/openai"
)

// retrieveFromOpenAI retrieves context using OpenAI's file_search tool
// This supports two modes:
// 1. Tool-based (Responses API workflow): Adds file_search tool to request, LLM calls it
// 2. Direct search: Uses vector store search API for synchronous retrieval
func (r *OpenAIRouter) retrieveFromOpenAI(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	openaiConfig, ok := ragConfig.BackendConfig.(*config.OpenAIRAGConfig)
	if !ok {
		return "", fmt.Errorf("invalid OpenAI RAG config")
	}

	baseURL := openaiConfig.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
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
		if err := r.addFileSearchToolToRequest(ctx, openaiConfig); err != nil {
			return "", fmt.Errorf("failed to add file_search tool: %w", err)
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
	searchResp, err := vectorStoreClient.SearchVectorStore(traceCtx, openaiConfig.VectorStoreID, query, limit, openaiConfig.Filter)
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
	if len(ctx.OriginalRequestBody) == 0 {
		return fmt.Errorf("original request body is empty")
	}

	// Parse the request body
	var requestMap map[string]interface{}
	if err := json.Unmarshal(ctx.OriginalRequestBody, &requestMap); err != nil {
		return fmt.Errorf("failed to parse request body: %w", err)
	}

	// Get or create tools array
	var tools []interface{}
	if existingTools, ok := requestMap["tools"].([]interface{}); ok {
		tools = existingTools
	} else {
		tools = make([]interface{}, 0)
	}

	// Check if file_search tool already exists
	for _, tool := range tools {
		toolMap, ok := tool.(map[string]interface{})
		if !ok {
			continue
		}
		if toolType, ok := toolMap["type"].(string); ok && toolType == "file_search" {
			// Tool already exists, update it
			logging.Debugf("file_search tool already exists, updating configuration")
			if toolConfig, ok := toolMap["file_search"].(map[string]interface{}); ok {
				// Update vector store IDs
				toolConfig["vector_store_ids"] = []string{openaiConfig.VectorStoreID}
				if openaiConfig.MaxNumResults != nil {
					toolConfig["max_num_results"] = *openaiConfig.MaxNumResults
				}
				if len(openaiConfig.FileIDs) > 0 {
					toolConfig["file_ids"] = openaiConfig.FileIDs
				}
				if openaiConfig.Filter != nil {
					toolConfig["filter"] = openaiConfig.Filter
				}
			}
			// Update the request body
			requestMap["tools"] = tools
			updatedBody, err := json.Marshal(requestMap)
			if err != nil {
				return fmt.Errorf("failed to marshal updated request: %w", err)
			}
			ctx.OriginalRequestBody = updatedBody
			return nil
		}
	}

	// Create file_search tool configuration
	fileSearchConfig := map[string]interface{}{
		"vector_store_ids": []string{openaiConfig.VectorStoreID},
	}

	if openaiConfig.MaxNumResults != nil {
		fileSearchConfig["max_num_results"] = *openaiConfig.MaxNumResults
	}
	if len(openaiConfig.FileIDs) > 0 {
		fileSearchConfig["file_ids"] = openaiConfig.FileIDs
	}
	if openaiConfig.Filter != nil {
		fileSearchConfig["filter"] = openaiConfig.Filter
	}

	// Add file_search tool
	fileSearchTool := map[string]interface{}{
		"type":        "file_search",
		"file_search": fileSearchConfig,
	}

	tools = append(tools, fileSearchTool)
	requestMap["tools"] = tools

	// Update the request body
	updatedBody, err := json.Marshal(requestMap)
	if err != nil {
		return fmt.Errorf("failed to marshal updated request: %w", err)
	}

	ctx.OriginalRequestBody = updatedBody
	logging.Infof("Added file_search tool to request (vector_store_id: %s)", openaiConfig.VectorStoreID)

	return nil
}

// NOTE: Tool-based workflow functions (handleFileSearchToolCall, extractContextFromFileSearchResults)
// are reserved for future implementation when response annotation extraction is added.
// For now, use "direct_search" workflow_mode for synchronous retrieval.
