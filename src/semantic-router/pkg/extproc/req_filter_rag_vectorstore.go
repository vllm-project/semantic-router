//go:build !windows

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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// recallFromVectorStore recalls documents from a local vector store using
// the router-owned vector-store runtime published through RuntimeRegistry.
func (r *OpenAIRouter) recallFromVectorStore(traceCtx context.Context, source *config.ContextSource, query string, recall config.MatchRecall) ([]match.Scored, error) {
	vsConfig, err := source.VectorStoreBackendConfig()
	if err != nil {
		return nil, fmt.Errorf("invalid vectorstore RAG config: %w", err)
	}
	if vsConfig.VectorStoreID == "" {
		return nil, fmt.Errorf("vector_store_id is required for vectorstore backend")
	}
	embedder := r.currentVectorStoreEmbedder()
	if embedder == nil {
		return nil, fmt.Errorf("embedder not initialized for vectorstore RAG")
	}
	manager := r.currentVectorStoreManager()
	if manager == nil {
		return nil, fmt.Errorf("vector store manager not initialized")
	}
	if err = manager.CheckEmbeddingCompatibility(vsConfig.VectorStoreID); err != nil {
		return nil, err
	}
	queryEmbedding, err := embedder.Embed(traceCtx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}
	results, err := manager.Search(traceCtx, vsConfig.VectorStoreID, queryEmbedding, recall.TopK, recallThreshold(recall), vectorStoreRAGFilter(query, vsConfig.FileIDs))
	if err != nil {
		return nil, fmt.Errorf("vectorstore search failed: %w", err)
	}
	texts := make([]string, len(results))
	similarities := make([]float64, len(results))
	for index, result := range results {
		texts[index], similarities[index] = result.Content, result.Score
	}
	logging.Debugf("RAG vectorstore: recalled %d chunks from store %s", len(results), vsConfig.VectorStoreID)
	return scoredDocuments(texts, similarities), nil
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
