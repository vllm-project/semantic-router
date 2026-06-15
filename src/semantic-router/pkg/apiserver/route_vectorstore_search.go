//go:build !windows && cgo

package apiserver

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// maxSearchResults caps the max_num_results parameter.
const maxSearchResults = 1000

// SearchRequest represents a vector store search request.
type SearchRequest struct {
	Query          string                          `json:"query"`
	MaxNumResults  int                             `json:"max_num_results,omitempty"`
	Filters        map[string]interface{}          `json:"filters,omitempty"`
	RankingOptions *RankingOptions                 `json:"ranking_options,omitempty"`
	Hybrid         *vectorstore.HybridSearchConfig `json:"hybrid,omitempty"`
}

// RankingOptions controls search result ranking.
type RankingOptions struct {
	ScoreThreshold float32 `json:"score_threshold,omitempty"`
}

type vectorStoreSearchParams struct {
	storeID   string
	request   SearchRequest
	topK      int
	threshold float32
}

func (s *ClassificationAPIServer) handleSearchVectorStore(w http.ResponseWriter, r *http.Request) {
	manager := s.currentVectorStoreManager()
	embedder := s.currentVectorStoreEmbedder()
	if manager == nil || embedder == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	// Backend search can be slow (e.g., Llama Stack embeds queries on CPU),
	// so extend the server's write deadline beyond the default 30s.
	rc := http.NewResponseController(w)
	_ = rc.SetWriteDeadline(time.Now().Add(180 * time.Second))

	params, err := s.parseVectorStoreSearchParams(r)
	if err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	queryEmbedding, err := embedder.Embed(params.request.Query)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "EMBEDDING_ERROR", "failed to generate query embedding")
		return
	}

	results, err := performVectorStoreSearch(r.Context(), manager, params, queryEmbedding)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "SEARCH_ERROR", "search failed")
		return
	}

	response := map[string]interface{}{
		"object": "vector_store.search_results.page",
		"data":   results,
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) parseVectorStoreSearchParams(r *http.Request) (vectorStoreSearchParams, error) {
	path := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
	storeID := strings.TrimSuffix(path, "/search")
	if storeID == "" || storeID == path {
		return vectorStoreSearchParams{}, fmt.Errorf("vector store ID is required")
	}

	var req SearchRequest
	if err := s.parseJSONRequestWithLimit(r, &req, maxVectorStoreJSONBodySize); err != nil {
		return vectorStoreSearchParams{}, err
	}
	if req.Query == "" {
		return vectorStoreSearchParams{}, fmt.Errorf("query is required")
	}

	req.Filters = ensureVectorStoreSearchFilters(req.Filters, req.Query)
	return vectorStoreSearchParams{
		storeID:   storeID,
		request:   req,
		topK:      normalizeVectorStoreSearchLimit(req.MaxNumResults),
		threshold: vectorStoreSearchThreshold(req.RankingOptions),
	}, nil
}

func normalizeVectorStoreSearchLimit(limit int) int {
	if limit <= 0 {
		return 10
	}
	if limit > maxSearchResults {
		return maxSearchResults
	}
	return limit
}

func vectorStoreSearchThreshold(options *RankingOptions) float32 {
	if options == nil {
		return 0
	}
	return options.ScoreThreshold
}

func ensureVectorStoreSearchFilters(filters map[string]interface{}, query string) map[string]interface{} {
	if filters == nil {
		filters = make(map[string]interface{})
	}

	// Llama Stack searches by text, not embedding. Pass the query text via
	// the filter map so it can use it. Other backends safely ignore this key.
	filters["_query_text"] = query
	return filters
}

func performVectorStoreSearch(
	ctx context.Context,
	manager *vectorstore.Manager,
	params vectorStoreSearchParams,
	queryEmbedding []float32,
) ([]vectorstore.SearchResult, error) {
	if params.request.Hybrid == nil {
		return manager.Backend().Search(
			ctx,
			params.storeID,
			queryEmbedding,
			params.topK,
			params.threshold,
			params.request.Filters,
		)
	}

	backend := manager.Backend()
	if searcher, ok := backend.(vectorstore.HybridSearcher); ok {
		return searcher.HybridSearch(
			ctx,
			params.storeID,
			params.request.Query,
			queryEmbedding,
			params.topK,
			params.threshold,
			params.request.Filters,
			params.request.Hybrid,
		)
	}

	return vectorstore.GenericHybridRerank(
		ctx,
		backend,
		params.storeID,
		params.request.Query,
		queryEmbedding,
		params.topK,
		params.threshold,
		params.request.Filters,
		params.request.Hybrid,
	)
}
