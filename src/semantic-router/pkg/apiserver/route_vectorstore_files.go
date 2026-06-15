//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// AttachFileRequest represents a request to attach a file to a vector store.
type AttachFileRequest struct {
	FileID           string                        `json:"file_id"`
	ChunkingStrategy *vectorstore.ChunkingStrategy `json:"chunking_strategy,omitempty"`
}

func (s *ClassificationAPIServer) handleAttachFile(w http.ResponseWriter, r *http.Request) {
	pipeline := s.currentVectorStorePipeline()
	if pipeline == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "VECTOR_STORE_DISABLED", "vector store feature is not enabled")
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/v1/vector_stores/")
	id := strings.TrimSuffix(path, "/files")
	if id == "" || id == path {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "vector store ID is required")
		return
	}

	var req AttachFileRequest
	if err := s.parseJSONRequestWithLimit(r, &req, maxVectorStoreJSONBodySize); err != nil {
		s.writeJSONRequestError(w, err)
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
