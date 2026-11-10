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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// recallFromVectorStore recalls documents from a local vector store using
// the router-owned vector-store runtime published through RuntimeRegistry:
// by the similarity of a query embedding, or by BM25 for bm25 recall, which
// embeds nothing.
func (r *OpenAIRouter) recallFromVectorStore(traceCtx context.Context, source *config.ContextSource, query string, recall config.MatchRecall) ([]match.Scored, error) {
	vsConfig, manager, err := r.ragVectorStore(source)
	if err != nil {
		return nil, err
	}
	filter := vectorStoreRAGFilter(query, vsConfig.FileIDs)
	var results []vectorstore.SearchResult
	if recall.EffectiveMethod() == config.MatchRecallBM25 {
		results, err = manager.LexicalSearch(traceCtx, vsConfig.VectorStoreID, query, recall.TopK, max(0, recallThreshold(recall)), filter)
	} else {
		results, err = r.searchVectorStore(traceCtx, manager, vsConfig.VectorStoreID, query, recall, filter)
	}
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

// searchVectorStore searches a store by the similarity of the query's
// embedding.
func (r *OpenAIRouter) searchVectorStore(traceCtx context.Context, manager *vectorstore.Manager, storeID, query string, recall config.MatchRecall, filter map[string]interface{}) ([]vectorstore.SearchResult, error) {
	embedder := r.currentVectorStoreEmbedder()
	if embedder == nil {
		return nil, fmt.Errorf("embedder not initialized for vectorstore RAG")
	}
	if err := manager.CheckEmbeddingCompatibility(storeID); err != nil {
		return nil, err
	}
	queryEmbedding, err := embedder.Embed(traceCtx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}
	return manager.Search(traceCtx, storeID, queryEmbedding, recall.TopK, recallThreshold(recall), filter)
}

// listVectorStoreDocuments lists a local vector store's chunks, the most
// recently inserted first, for a rag source whose judge reads the store
// without recall.
func (r *OpenAIRouter) listVectorStoreDocuments(traceCtx context.Context, source *config.ContextSource, limit int) ([]match.Candidate, int, error) {
	vsConfig, manager, err := r.ragVectorStore(source)
	if err != nil {
		return nil, 0, err
	}
	results, held, err := manager.ListChunks(traceCtx, vsConfig.VectorStoreID, limit, vectorStoreRAGFilter("", vsConfig.FileIDs))
	if err != nil {
		return nil, 0, fmt.Errorf("vectorstore listing failed: %w", err)
	}
	texts := make([]string, len(results))
	for index, result := range results {
		texts[index] = result.Content
	}
	documents := scoredDocuments(texts, nil)
	candidates := make([]match.Candidate, len(documents))
	for index, document := range documents {
		candidates[index] = document.Candidate
	}
	return candidates, held - (len(results) - len(candidates)), nil
}

// ragVectorStore is a vectorstore source's configuration and the manager of
// the router's vector store.
func (r *OpenAIRouter) ragVectorStore(source *config.ContextSource) (*config.VectorStoreRAGConfig, *vectorstore.Manager, error) {
	vsConfig, err := source.VectorStoreBackendConfig()
	if err != nil {
		return nil, nil, fmt.Errorf("invalid vectorstore RAG config: %w", err)
	}
	if vsConfig.VectorStoreID == "" {
		return nil, nil, fmt.Errorf("vector_store_id is required for vectorstore backend")
	}
	manager := r.currentVectorStoreManager()
	if manager == nil {
		return nil, nil, fmt.Errorf("vector store manager not initialized")
	}
	return vsConfig, manager, nil
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
