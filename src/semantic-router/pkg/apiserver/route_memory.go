//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// safeIDPattern allows only alphanumeric chars and common ID separators.
// Rejects characters that could manipulate Milvus filter expressions (e.g. ", \, ||, &&).
var safeIDPattern = regexp.MustCompile(`^[a-zA-Z0-9._@:/$-]+$`)

// validMemoryTypes is the set of accepted memory type values
var validMemoryTypes = map[memory.MemoryType]bool{
	memory.MemoryTypeSemantic:   true,
	memory.MemoryTypeProcedural: true,
	memory.MemoryTypeEpisodic:   true,
}

// MemoryResponse wraps a single memory for API responses
type MemoryResponse struct {
	ID          string            `json:"id"`
	Type        memory.MemoryType `json:"type"`
	Content     string            `json:"content"`
	UserID      string            `json:"user_id"`
	Source      string            `json:"source,omitempty"`
	Importance  float32           `json:"importance"`
	AccessCount int               `json:"access_count"`
	CreatedAt   string            `json:"created_at"`
	UpdatedAt   string            `json:"updated_at,omitempty"`
}

// MemoryListResponse wraps a list of memories with total count
type MemoryListResponse struct {
	Memories []MemoryResponse `json:"memories"`
	Total    int              `json:"total"`
	Limit    int              `json:"limit"`
}

// MemoryDeleteResponse represents the response from a delete operation
type MemoryDeleteResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

// requireMemoryStore checks if the memory store is available and returns an error if not.
// Returns true if the store is available, false otherwise.
func (s *ClassificationAPIServer) requireMemoryStore(w http.ResponseWriter) bool {
	if s.memoryStore == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "MEMORY_NOT_AVAILABLE",
			"Memory store is not configured or not yet initialized. Enable memory in configuration.")
		return false
	}
	return true
}

// extractUserID extracts the user_id with priority: auth header > query param fallback.
//
// Priority 1: x-authz-user-id header injected by the external auth service
// (Authorino, Envoy Gateway JWT, oauth2-proxy, etc.). This is the trusted source.
//
// Priority 2: user_id query parameter. This is untrusted (client-provided)
// and intended for development/testing without a full auth stack.
func (s *ClassificationAPIServer) extractUserID(w http.ResponseWriter, r *http.Request) (string, bool) {
	var userID string

	// Check auth header first (trusted source, injected by auth backend)
	if h := r.Header.Get(headers.AuthzUserID); h != "" {
		userID = h
	} else if q := r.URL.Query().Get("user_id"); q != "" {
		// Fallback to query parameter (untrusted, for development/testing)
		userID = q
	} else {
		s.writeErrorResponse(w, http.StatusUnauthorized, "MISSING_USER_ID",
			"User identity required. Set the auth header (x-authz-user-id) via your auth layer, "+
				"or user_id query parameter for development")
		return "", false
	}

	// Validate against injection characters (defense-in-depth for both trusted and untrusted sources)
	if !safeIDPattern.MatchString(userID) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_USER_ID",
			"user_id contains invalid characters")
		return "", false
	}

	return userID, true
}

// extractMemoryID extracts and validates the memory ID from the URL path.
// Rejects IDs containing characters that could manipulate Milvus filter expressions.
func (s *ClassificationAPIServer) extractMemoryID(w http.ResponseWriter, r *http.Request) (string, bool) {
	memoryID := r.PathValue("id")
	if memoryID == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "MISSING_ID", "memory ID is required in path")
		return "", false
	}
	if !safeIDPattern.MatchString(memoryID) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_ID",
			"memory ID contains invalid characters")
		return "", false
	}
	return memoryID, true
}

// parseMemoryTypes parses and validates a comma-separated type filter string.
// Returns validated types and true, or writes an error response and returns false.
func (s *ClassificationAPIServer) parseMemoryTypes(w http.ResponseWriter, typeStr string) ([]memory.MemoryType, bool) {
	if typeStr == "" {
		return nil, true
	}

	var types []memory.MemoryType
	for _, t := range strings.Split(typeStr, ",") {
		trimmed := strings.TrimSpace(t)
		if trimmed == "" {
			continue
		}
		mt := memory.MemoryType(trimmed)
		if !validMemoryTypes[mt] {
			s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_TYPE",
				"Invalid memory type: "+trimmed+". Valid types: semantic, procedural, episodic")
			return nil, false
		}
		types = append(types, mt)
	}
	return types, true
}

// memoryToResponse converts a Memory to the API response format
func memoryToResponse(mem *memory.Memory) MemoryResponse {
	resp := MemoryResponse{
		ID:          mem.ID,
		Type:        mem.Type,
		Content:     mem.Content,
		UserID:      mem.UserID,
		Source:      mem.Source,
		Importance:  mem.Importance,
		AccessCount: mem.AccessCount,
		CreatedAt:   mem.CreatedAt.UTC().Format(time.RFC3339),
	}
	if !mem.UpdatedAt.IsZero() {
		resp.UpdatedAt = mem.UpdatedAt.UTC().Format(time.RFC3339)
	}
	return resp
}

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

	// Parse and validate memory type filter
	types, ok := s.parseMemoryTypes(w, r.URL.Query().Get("type"))
	if !ok {
		return
	}
	opts.Types = types

	// Parse limit
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if limit, err := strconv.Atoi(limitStr); err == nil {
			opts.Limit = limit
		}
	}

	ctx := r.Context()
	result, err := s.memoryStore.List(ctx, opts)
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
	mem, err := s.memoryStore.Get(ctx, memoryID)
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
	mem, err := s.memoryStore.Get(ctx, memoryID)
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

	if err := s.memoryStore.Forget(ctx, memoryID); err != nil {
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
	if err := s.memoryStore.ForgetByScope(ctx, scope); err != nil {
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
