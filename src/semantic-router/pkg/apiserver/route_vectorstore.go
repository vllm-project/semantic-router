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
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// maxVectorStoreJSONBodySize limits vector-store JSON requests that should not
// carry raw file content.
const maxVectorStoreJSONBodySize int64 = 1 * 1024 * 1024

// SetVectorStoreManager sets the global vector store manager for the API server.
func SetVectorStoreManager(mgr *vectorstore.Manager) {
	globalRuntimeDeps.setVectorStoreManager(mgr)
}

// SetIngestionPipeline sets the global ingestion pipeline for the API server.
func SetIngestionPipeline(p *vectorstore.IngestionPipeline) {
	globalRuntimeDeps.setIngestionPipeline(p)
}

// SetEmbedder sets the global embedder for search queries.
func SetEmbedder(e vectorstore.Embedder) {
	globalRuntimeDeps.setEmbedder(e)
}

// GetEmbedder returns the global embedder instance.
func GetEmbedder() vectorstore.Embedder {
	return globalRuntimeDeps.getEmbedder()
}

// GetVectorStoreManager returns the global vector store manager instance.
func GetVectorStoreManager() *vectorstore.Manager {
	return globalRuntimeDeps.getVectorStoreManager()
}

func (s *ClassificationAPIServer) handleCreateVectorStore(w http.ResponseWriter, r *http.Request) {
	manager := s.currentVectorStoreManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	var req vectorstore.CreateStoreRequest
	if err := s.parseJSONRequestWithLimit(r, &req, maxVectorStoreJSONBodySize); err != nil {
		s.writeJSONRequestError(w, err)
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

	params, ok := s.parseVectorStoreListParams(w, r)
	if !ok {
		return
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

	var req vectorstore.UpdateStoreRequest
	if err := s.parseJSONRequestWithLimit(r, &req, maxVectorStoreJSONBodySize); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	vs, err := manager.UpdateStore(r.Context(), id, req)
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

// extractPathParam extracts the ID parameter from a URL path after the given prefix.
func extractPathParam(path, prefix string) string {
	trimmed := strings.TrimPrefix(path, prefix)
	// Remove any trailing slash or sub-path.
	if idx := strings.Index(trimmed, "/"); idx != -1 {
		trimmed = trimmed[:idx]
	}
	return trimmed
}
