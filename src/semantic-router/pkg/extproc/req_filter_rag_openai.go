package extproc

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/openai"
)

// recallFromOpenAI recalls documents with the vector store search API of an
// OpenAI-compatible service, which returns at most max_num_results (20 by
// default) with their scores. The Responses API's server-side file search
// (workflow_mode tool_based) is not available on this backend interface.
func (r *OpenAIRouter) recallFromOpenAI(traceCtx context.Context, source *config.ContextSource, query string, _ config.MatchRecall) ([]match.Scored, error) {
	openaiConfig, err := source.OpenAIBackendConfig()
	if err != nil {
		return nil, fmt.Errorf("invalid OpenAI RAG config: %w", err)
	}
	if openaiConfig.WorkflowMode == "tool_based" {
		return nil, fmt.Errorf("failed to add file_search tool: %w", r.addFileSearchToolToRequest(openaiConfig))
	}
	baseURL := openaiConfig.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	logging.Infof("OpenAI RAG: Using direct search workflow (vector_store_id: %s)", openaiConfig.VectorStoreID)
	vectorStoreClient := openai.NewVectorStoreClientWithTimeout(
		baseURL,
		openaiConfig.APIKey,
		openaiConfig.MaxResponseBytes,
		openaiConfig.GetTimeout(),
	)
	limit := 20
	if openaiConfig.MaxNumResults != nil {
		limit = *openaiConfig.MaxNumResults
	}
	filterMap, err := openaiConfig.FilterMap()
	if err != nil {
		return nil, fmt.Errorf("invalid OpenAI filter config: %w", err)
	}
	searchResp, err := vectorStoreClient.SearchVectorStore(traceCtx, openaiConfig.VectorStoreID, query, limit, filterMap)
	if err != nil {
		return nil, fmt.Errorf("vector store search failed: %w", err)
	}
	texts := make([]string, len(searchResp.Data))
	similarities := make([]float64, len(searchResp.Data))
	for index, result := range searchResp.Data {
		texts[index], similarities[index] = result.Content, result.Score
	}
	logging.Infof("Recalled %d documents from OpenAI vector store (vector_store_id: %s)", len(texts), openaiConfig.VectorStoreID)
	return scoredDocuments(texts, similarities), nil
}

// addFileSearchToolToRequest adds the file_search tool to the request
// This follows the Responses API workflow where tools are part of the request
func (r *OpenAIRouter) addFileSearchToolToRequest(_ *config.OpenAIRAGConfig) error {
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
