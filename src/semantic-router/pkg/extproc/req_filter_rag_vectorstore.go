//go:build !windows && cgo

/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// retrieveFromVectorStore retrieves context from a local vector store.
// It uses the global embedder and vector store manager set via apiserver.
func (r *OpenAIRouter) retrieveFromVectorStore(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	params, err := resolveVectorStoreRetrievalParams(ctx, ragConfig)
	if err != nil {
		return "", err
	}

	embedder := r.currentVectorStoreEmbedder()
	if embedder == nil {
		return "", fmt.Errorf("embedder not initialized for vectorstore RAG")
	}

	queryEmbedding, err := embedder.Embed(params.query)
	if err != nil {
		return "", fmt.Errorf("failed to generate query embedding: %w", err)
	}

	manager := r.currentVectorStoreManager()
	if manager == nil {
		return "", fmt.Errorf("vector store manager not initialized")
	}

	results, err := manager.Backend().Search(
		traceCtx,
		params.storeID,
		queryEmbedding,
		params.topK,
		params.threshold,
		params.filter,
	)
	if err != nil {
		return "", fmt.Errorf("vectorstore search failed: %w", err)
	}

	retrievedContext, bestScore, found := formatVectorStoreRetrievalResults(results)
	if !found {
		logging.Debugf("RAG vectorstore: no results found for query in store %s", params.storeID)
		return "", nil
	}

	// Store best similarity score for observability.
	ctx.RAGSimilarityScore = bestScore

	logging.Debugf("RAG vectorstore: retrieved %d chunks from store %s (best score: %.4f)",
		len(results), params.storeID, bestScore)

	return retrievedContext, nil
}

type vectorStoreRetrievalParams struct {
	storeID   string
	query     string
	topK      int
	threshold float32
	filter    map[string]interface{}
}

func resolveVectorStoreRetrievalParams(
	ctx *RequestContext,
	ragConfig *config.RAGPluginConfig,
) (vectorStoreRetrievalParams, error) {
	vsConfig, err := ragConfig.VectorStoreBackendConfig()
	if err != nil {
		return vectorStoreRetrievalParams{}, fmt.Errorf("invalid vectorstore RAG config: %w", err)
	}
	if vsConfig == nil {
		return vectorStoreRetrievalParams{}, fmt.Errorf("invalid vectorstore RAG config: missing backend config")
	}
	if vsConfig.VectorStoreID == "" {
		return vectorStoreRetrievalParams{}, fmt.Errorf("vector_store_id is required for vectorstore backend")
	}

	query := ctx.UserContent
	if query == "" {
		return vectorStoreRetrievalParams{}, fmt.Errorf("no user content for RAG retrieval")
	}

	return vectorStoreRetrievalParams{
		storeID:   vsConfig.VectorStoreID,
		query:     query,
		topK:      vectorStoreRAGTopK(ragConfig),
		threshold: vectorStoreRAGThreshold(ragConfig),
		filter:    vectorStoreRAGFilter(query, vsConfig.FileIDs),
	}, nil
}

func vectorStoreRAGTopK(ragConfig *config.RAGPluginConfig) int {
	if ragConfig.TopK == nil {
		return 5
	}
	return *ragConfig.TopK
}

func vectorStoreRAGThreshold(ragConfig *config.RAGPluginConfig) float32 {
	if ragConfig.SimilarityThreshold == nil {
		return 0.7
	}
	return *ragConfig.SimilarityThreshold
}

func vectorStoreRAGFilter(query string, fileIDs []string) map[string]interface{} {
	// Llama Stack searches by text, not embedding. Pass the query text via
	// the filter map so it can use it.
	filter := map[string]interface{}{
		"_query_text": query,
	}
	if len(fileIDs) > 0 {
		filter["file_id"] = fileIDs[0]
	}
	return filter
}

func formatVectorStoreRetrievalResults(results []vectorstore.SearchResult) (string, float32, bool) {
	if len(results) == 0 {
		return "", 0, false
	}

	parts := make([]string, 0, len(results))
	for _, result := range results {
		parts = append(parts, result.Content)
	}

	return strings.Join(parts, "\n\n---\n\n"), float32(results[0].Score), true
}
