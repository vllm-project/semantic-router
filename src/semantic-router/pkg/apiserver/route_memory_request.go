//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

const maxMemoryListLimit = 100

// safeIDPattern allows only alphanumeric chars and common ID separators.
// Rejects characters that could manipulate store filter expressions.
var safeIDPattern = regexp.MustCompile(`^[a-zA-Z0-9._@:/$-]+$`)

var validMemoryTypes = map[memory.MemoryType]bool{
	memory.MemoryTypeSemantic:   true,
	memory.MemoryTypeProcedural: true,
	memory.MemoryTypeEpisodic:   true,
}

// requireMemoryStore returns false after writing an error response when memory is unavailable.
func (s *ClassificationAPIServer) requireMemoryStore(w http.ResponseWriter) bool {
	if s.currentMemoryStore() == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "MEMORY_NOT_AVAILABLE",
			"Memory store is not configured or not yet initialized. Enable memory in configuration.")
		return false
	}
	return true
}

// extractUserID extracts the user_id with priority: auth header > query param fallback.
//
// Priority 1: x-authz-user-id header injected by the external auth service.
// Priority 2: user_id query parameter for development/testing without a full auth stack.
func (s *ClassificationAPIServer) extractUserID(w http.ResponseWriter, r *http.Request) (string, bool) {
	var userID string
	if h := r.Header.Get(headers.AuthzUserID); h != "" {
		userID = h
	} else if q := r.URL.Query().Get("user_id"); q != "" {
		userID = q
	} else {
		s.writeErrorResponse(w, http.StatusUnauthorized, "MISSING_USER_ID",
			"User identity required. Set the auth header (x-authz-user-id) via your auth layer, "+
				"or user_id query parameter for development")
		return "", false
	}

	if !safeIDPattern.MatchString(userID) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_USER_ID",
			"user_id contains invalid characters")
		return "", false
	}

	return userID, true
}

// extractMemoryID extracts and validates the memory ID from the URL path.
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

func (s *ClassificationAPIServer) parseMemoryListLimit(w http.ResponseWriter, limitStr string) (int, bool) {
	if limitStr == "" {
		return 0, true
	}

	limit, err := strconv.Atoi(limitStr)
	if err != nil || limit <= 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_LIMIT",
			"limit must be a positive integer")
		return 0, false
	}
	if limit > maxMemoryListLimit {
		return maxMemoryListLimit, true
	}
	return limit, true
}
