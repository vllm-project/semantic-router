//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// handleListMemories handles GET /v1/memory
// Lists memories for a user with optional filtering.
// Returns up to `limit` most recent memories sorted by created_at descending.
//
// User identity: x-authz-user-id header (trusted) or user_id query param (dev fallback)
//
// Query parameters:
//   - type: filter by memory type (semantic, procedural, episodic)
//   - limit: max results (default 20, max 100)
func (s *ClassificationAPIServer) handleListMemories(w http.ResponseWriter, r *http.Request) {
	if !s.requireMemoryStore(w) {
		return
	}

	userID, ok := s.extractUserID(w, r)
	if !ok {
		return
	}

	opts := memory.ListOptions{
		UserID: userID,
	}

	types, ok := s.parseMemoryTypes(w, r.URL.Query().Get("type"))
	if !ok {
		return
	}
	opts.Types = types

	limit, ok := s.parseMemoryListLimit(w, r.URL.Query().Get("limit"))
	if !ok {
		return
	}
	opts.Limit = limit

	ctx := r.Context()
	result, err := s.currentMemoryStore().List(ctx, opts)
	if err != nil {
		logging.Errorf("[MemoryAPI] List failed for user_id=%s: %v", userID, err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "LIST_FAILED",
			"Failed to list memories")
		return
	}

	// Convert to response format
	memories := make([]MemoryResponse, 0, len(result.Memories))
	for _, mem := range result.Memories {
		memories = append(memories, memoryToResponse(mem))
	}

	response := MemoryListResponse{
		Memories: memories,
		Total:    result.Total,
		Limit:    result.Limit,
	}

	logging.Debugf("[MemoryAPI] Listed %d/%d memories for user_id=%s", len(memories), result.Total, userID)
	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleGetMemory handles GET /v1/memory/{id}
// Retrieves a specific memory by ID, enforcing ownership via authenticated user identity.
func (s *ClassificationAPIServer) handleGetMemory(w http.ResponseWriter, r *http.Request) {
	if !s.requireMemoryStore(w) {
		return
	}

	memoryID, ok := s.extractMemoryID(w, r)
	if !ok {
		return
	}

	userID, ok := s.extractUserID(w, r)
	if !ok {
		return
	}

	ctx := r.Context()
	mem, err := s.currentMemoryStore().Get(ctx, memoryID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND",
				"Memory not found: "+memoryID)
			return
		}
		logging.Errorf("[MemoryAPI] Get failed for id=%s: %v", memoryID, err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "GET_FAILED",
			"Failed to retrieve memory")
		return
	}

	// Enforce user ownership - user can only access their own memories
	if mem.UserID != userID {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND",
			"Memory not found: "+memoryID)
		return
	}

	logging.Debugf("[MemoryAPI] Retrieved memory id=%s for user_id=%s", memoryID, userID)
	s.writeJSONResponse(w, http.StatusOK, memoryToResponse(mem))
}

// handleDeleteMemory handles DELETE /v1/memory/{id}
// Deletes a specific memory by ID, enforcing ownership via authenticated user identity.
func (s *ClassificationAPIServer) handleDeleteMemory(w http.ResponseWriter, r *http.Request) {
	if !s.requireMemoryStore(w) {
		return
	}

	memoryID, ok := s.extractMemoryID(w, r)
	if !ok {
		return
	}

	userID, ok := s.extractUserID(w, r)
	if !ok {
		return
	}

	ctx := r.Context()

	// Verify ownership before deleting
	mem, err := s.currentMemoryStore().Get(ctx, memoryID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND",
				"Memory not found: "+memoryID)
			return
		}
		logging.Errorf("[MemoryAPI] Get failed during delete for id=%s: %v", memoryID, err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "DELETE_FAILED",
			"Failed to delete memory")
		return
	}

	if mem.UserID != userID {
		s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND",
			"Memory not found: "+memoryID)
		return
	}

	if err := s.currentMemoryStore().Forget(ctx, memoryID); err != nil {
		// Handle TOCTOU: if another request deleted this memory between Get and Forget,
		// treat it as a successful idempotent delete rather than a 500.
		if strings.Contains(err.Error(), "not found") {
			logging.Debugf("[MemoryAPI] Memory id=%s already deleted (concurrent request)", memoryID)
		} else {
			logging.Errorf("[MemoryAPI] Delete failed for id=%s: %v", memoryID, err)
			s.writeErrorResponse(w, http.StatusInternalServerError, "DELETE_FAILED",
				"Failed to delete memory")
			return
		}
	}

	logging.Infof("[MemoryAPI] Deleted memory id=%s for user_id=%s", memoryID, userID)
	s.writeJSONResponse(w, http.StatusOK, MemoryDeleteResponse{
		Success: true,
		Message: "Memory deleted successfully",
	})
}

// handleDeleteMemoriesByScope handles DELETE /v1/memory[?type=semantic]
// Deletes all memories for the authenticated user, optionally filtered by type.
func (s *ClassificationAPIServer) handleDeleteMemoriesByScope(w http.ResponseWriter, r *http.Request) {
	if !s.requireMemoryStore(w) {
		return
	}

	userID, ok := s.extractUserID(w, r)
	if !ok {
		return
	}

	scope := memory.MemoryScope{
		UserID: userID,
	}

	// Parse and validate type filter
	types, ok := s.parseMemoryTypes(w, r.URL.Query().Get("type"))
	if !ok {
		return
	}
	scope.Types = types

	ctx := r.Context()
	if err := s.currentMemoryStore().ForgetByScope(ctx, scope); err != nil {
		logging.Errorf("[MemoryAPI] DeleteByScope failed for user_id=%s: %v", userID, err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "DELETE_FAILED",
			"Failed to delete memories")
		return
	}

	scopeDesc := "all memories"
	if len(scope.Types) > 0 {
		typeStrs := make([]string, 0, len(scope.Types))
		for _, t := range scope.Types {
			typeStrs = append(typeStrs, string(t))
		}
		scopeDesc = "memories of type: " + strings.Join(typeStrs, ", ")
	}

	logging.Infof("[MemoryAPI] Deleted %s for user_id=%s", scopeDesc, userID)
	s.writeJSONResponse(w, http.StatusOK, MemoryDeleteResponse{
		Success: true,
		Message: "Successfully deleted " + scopeDesc,
	})
}
