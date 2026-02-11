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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// retrieveFromVectorStore retrieves context from a local vector store.
// It uses the global embedder and vector store manager set via apiserver.
func (r *OpenAIRouter) retrieveFromVectorStore(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	// Extract vectorstore-specific config.
	vsConfig, ok := ragConfig.BackendConfig.(*config.VectorStoreRAGConfig)
	if !ok || vsConfig == nil {
		return "", fmt.Errorf("invalid vectorstore RAG config: BackendConfig must be *VectorStoreRAGConfig")
	}

	if vsConfig.VectorStoreID == "" {
		return "", fmt.Errorf("vector_store_id is required for vectorstore backend")
	}

	query := ctx.UserContent
	if query == "" {
		return "", fmt.Errorf("no user content for RAG retrieval")
	}

	// Get search parameters with defaults.
	topK := 5
	if ragConfig.TopK != nil {
		topK = *ragConfig.TopK
	}

	threshold := float32(0.7)
	if ragConfig.SimilarityThreshold != nil {
		threshold = *ragConfig.SimilarityThreshold
	}

	// Generate query embedding using the global embedder.
	embedder := apiserver.GetEmbedder()
	if embedder == nil {
		return "", fmt.Errorf("embedder not initialized for vectorstore RAG")
	}

	queryEmbedding, err := embedder.Embed(query)
	if err != nil {
		return "", fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Get the vector store manager.
	manager := apiserver.GetVectorStoreManager()
	if manager == nil {
		return "", fmt.Errorf("vector store manager not initialized")
	}

	// Build filter.
	var filter map[string]interface{}
	if len(vsConfig.FileIDs) > 0 {
		// If specific file IDs are configured, filter to those.
		filter = map[string]interface{}{"file_id": vsConfig.FileIDs[0]}
	}

	// Search.
	results, err := manager.Backend().Search(traceCtx, vsConfig.VectorStoreID, queryEmbedding, topK, threshold, filter)
	if err != nil {
		return "", fmt.Errorf("vectorstore search failed: %w", err)
	}

	if len(results) == 0 {
		logging.Debugf("RAG vectorstore: no results found for query in store %s", vsConfig.VectorStoreID)
		return "", nil
	}

	// Combine results into context string.
	var parts []string
	for _, result := range results {
		parts = append(parts, result.Content)
	}

	retrievedContext := strings.Join(parts, "\n\n---\n\n")

	// Store best similarity score for observability.
	if len(results) > 0 {
		ctx.RAGSimilarityScore = float32(results[0].Score)
	}

	logging.Debugf("RAG vectorstore: retrieved %d chunks from store %s (best score: %.4f)",
		len(results), vsConfig.VectorStoreID, results[0].Score)

	return retrievedContext, nil
}
