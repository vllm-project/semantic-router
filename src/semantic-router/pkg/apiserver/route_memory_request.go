//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
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

// extractUserID extracts the ingress identity used to scope management memory
// operations. Query parameters and request-body fields are never identity
// sources because callers can author them directly.
func (s *ClassificationAPIServer) extractUserID(w http.ResponseWriter, r *http.Request) (string, bool) {
	identity := s.extractTrustedIdentity(r)
	userID := strings.TrimSpace(identity.UserID)
	if userID == "" {
		s.writeErrorResponse(w, http.StatusUnauthorized, "MISSING_USER_ID",
			"User identity required. Set the configured auth identity header via your auth layer")
		return "", false
	}

	if !safeIDPattern.MatchString(userID) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_USER_ID",
			"user_id contains invalid characters")
		return "", false
	}

	return userID, true
}

func (s *ClassificationAPIServer) extractTrustedIdentity(r *http.Request) authz.TrustedIdentity {
	if r == nil {
		return authz.TrustedIdentity{}
	}
	headerName := headers.AuthzUserID
	if cfg := s.currentConfig(); cfg != nil {
		headerName = cfg.Authz.Identity.GetUserIDHeader()
	}
	return authz.TrustedIdentity{UserID: requestHeaderValueCI(r.Header, headerName)}
}

func requestHeaderValueCI(header http.Header, name string) string {
	if name == "" {
		return ""
	}
	if value := header.Get(name); strings.TrimSpace(value) != "" {
		return value
	}
	for key, values := range header {
		if !strings.EqualFold(key, name) {
			continue
		}
		for _, value := range values {
			if strings.TrimSpace(value) != "" {
				return value
			}
		}
	}
	return ""
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
