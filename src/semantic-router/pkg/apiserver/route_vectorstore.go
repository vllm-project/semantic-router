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

package apiserver

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// maxJSONBodySize limits the request body for JSON endpoints (1 MB).
const maxJSONBodySize = 1 * 1024 * 1024

// maxSearchResults caps the max_num_results parameter.
const maxSearchResults = 1000

// limitBody applies a size limit to the request body to prevent resource exhaustion.
func limitBody(r *http.Request) {
	r.Body = http.MaxBytesReader(nil, r.Body, maxJSONBodySize)
}

// vectorStoreManager is the global vector store manager instance.
// It is set during initialization via SetVectorStoreManager.
var vectorStoreManager *vectorstore.Manager

// globalPipeline is the global ingestion pipeline instance.
var globalPipeline *vectorstore.IngestionPipeline

// globalEmbedder is the global embedder instance for search queries.
var globalEmbedder vectorstore.Embedder

// SetVectorStoreManager sets the global vector store manager for the API server.
func SetVectorStoreManager(mgr *vectorstore.Manager) {
	vectorStoreManager = mgr
}

// SetIngestionPipeline sets the global ingestion pipeline for the API server.
func SetIngestionPipeline(p *vectorstore.IngestionPipeline) {
	globalPipeline = p
}

// SetEmbedder sets the global embedder for search queries.
func SetEmbedder(e vectorstore.Embedder) {
	globalEmbedder = e
}

// GetEmbedder returns the global embedder instance.
func GetEmbedder() vectorstore.Embedder {
	return globalEmbedder
}

// GetVectorStoreManager returns the global vector store manager instance.
func GetVectorStoreManager() *vectorstore.Manager {
	return vectorStoreManager
}

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

// AttachFileRequest represents a request to attach a file to a vector store.
type AttachFileRequest struct {
	FileID           string                        `json:"file_id"`
	ChunkingStrategy *vectorstore.ChunkingStrategy `json:"chunking_strategy,omitempty"`
}

// registerVectorStoreRoutes registers vector store management routes.
func registerVectorStoreRoutes(mux *http.ServeMux, s *ClassificationAPIServer) {
	mux.HandleFunc("POST /v1/vector_stores", s.handleCreateVectorStore)
	mux.HandleFunc("GET /v1/vector_stores", s.handleListVectorStores)
	mux.HandleFunc("GET /v1/vector_stores/{id}", s.handleGetVectorStore)
	mux.HandleFunc("POST /v1/vector_stores/{id}", s.handleUpdateVectorStore)
	mux.HandleFunc("DELETE /v1/vector_stores/{id}", s.handleDeleteVectorStore)
	mux.HandleFunc("POST /v1/vector_stores/{id}/search", s.handleSearchVectorStore)
	mux.HandleFunc("POST /v1/vector_stores/{id}/files", s.handleAttachFile)
	mux.HandleFunc("GET /v1/vector_stores/{id}/files", s.handleListVectorStoreFiles)
	mux.HandleFunc("DELETE /v1/vector_stores/{id}/files/{file_id}", s.handleDetachFile)
}

func (s *ClassificationAPIServer) handleCreateVectorStore(w http.ResponseWriter, r *http.Request) {
	manager := s.currentVectorStoreManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	limitBody(r)
	var req vectorstore.CreateStoreRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	vs, err := manager.CreateStore(r.Context(), req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CREATE_FAILED", "failed to create vector store")
		return
	}

	s.writeJSONResponse(w, http.StatusOK, vs)
}

func (s *ClassificationAPIServer) handleListVectorStores(w http.ResponseWriter, r *http.Request) {
	manager := s.currentVectorStoreManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	query := r.URL.Query()
	limit, _ := strconv.Atoi(query.Get("limit"))
	params := vectorstore.ListStoresParams{
		Limit:  limit,
		Order:  query.Get("order"),
		After:  query.Get("after"),
		Before: query.Get("before"),
	}

	stores := manager.ListStores(params)

	response := map[string]interface{}{
		"object": "list",
		"data":   stores,
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleGetVectorStore(w http.ResponseWriter, r *http.Request) {
	manager := s.currentVectorStoreManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	id := extractPathParam(r.URL.Path, "/v1/vector_stores/")
	if id == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "vector store ID is required")
		return
	}

	vs, err := manager.GetStore(id)
	if err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, vs)
}

func (s *ClassificationAPIServer) handleUpdateVectorStore(w http.ResponseWriter, r *http.Request) {
	manager := s.currentVectorStoreManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	id := extractPathParam(r.URL.Path, "/v1/vector_stores/")
	if id == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "vector store ID is required")
		return
	}

	limitBody(r)
	var req vectorstore.UpdateStoreRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	vs, err := manager.UpdateStore(id, req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", "vector store not found")
		return
	}

	s.writeJSONResponse(w, http.StatusOK, vs)
}

func (s *ClassificationAPIServer) handleDeleteVectorStore(w http.ResponseWriter, r *http.Request) {
	manager := s.currentVectorStoreManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	id := extractPathParam(r.URL.Path, "/v1/vector_stores/")
	if id == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "vector store ID is required")
		return
	}

	if err := manager.DeleteStore(r.Context(), id); err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"id":      id,
		"object":  "vector_store.deleted",
		"deleted": true,
	})
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
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Generate query embedding.
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

	limitBody(r)

	var req SearchRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
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
		// Native hybrid search (e.g. in-memory backend with full-collection BM25/n-gram indexes).
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

	// Generic fallback: fetch expanded vector candidates, then re-rank with BM25 + n-gram.
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

func (s *ClassificationAPIServer) handleAttachFile(w http.ResponseWriter, r *http.Request) {
	pipeline := s.currentVectorStorePipeline()
	if pipeline == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	// Extract vector store ID from /v1/vector_stores/{id}/files
	path := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
	id := strings.TrimSuffix(path, "/files")
	if id == "" || id == path {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "vector store ID is required")
		return
	}

	limitBody(r)
	var req AttachFileRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	if req.FileID == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "file_id is required")
		return
	}

	vsf, err := pipeline.AttachFile(id, req.FileID, req.ChunkingStrategy)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "ATTACH_ERROR", "failed to attach file")
		return
	}

	s.writeJSONResponse(w, http.StatusOK, vsf)
}

func (s *ClassificationAPIServer) handleListVectorStoreFiles(w http.ResponseWriter, r *http.Request) {
	pipeline := s.currentVectorStorePipeline()
	if pipeline == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	// Extract vector store ID from /v1/vector_stores/{id}/files
	path := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
	id := strings.TrimSuffix(path, "/files")
	if id == "" || id == path {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "vector store ID is required")
		return
	}

	files := pipeline.ListFileStatuses(id)

	response := map[string]interface{}{
		"object": "list",
		"data":   files,
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleDetachFile(w http.ResponseWriter, r *http.Request) {
	pipeline := s.currentVectorStorePipeline()
	if pipeline == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	// Extract IDs from /v1/vector_stores/{id}/files/{file_id}
	path := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
	parts := strings.SplitN(path, "/files/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "vector store ID and file ID are required")
		return
	}

	storeID := parts[0]
	vsfID := parts[1]

	if err := pipeline.DetachFile(r.Context(), storeID, vsfID); err != nil {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"id":      vsfID,
		"object":  "vector_store.file.deleted",
		"deleted": true,
	})
}

// extractPathParam extracts the ID parameter from a URL path after the given prefix.
func extractPathParam(path, prefix string) string {
	trimmed := strings.TrimPrefix(path, prefix)
	// Remove any trailing slash or sub-path.
	if idx := strings.Index(trimmed, "/"); idx != -1 {
		trimmed = trimmed[:idx]
	}
	return trimmed
}
