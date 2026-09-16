//go:build !windows && cgo

package apiserver

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strconv"
	"time"
)

const maxManagementAuditEntries = 10000

type managementAuditEntry struct {
	Sequence     uint64           `json:"sequence"`
	Timestamp    string           `json:"timestamp"`
	Action       RouteAuditAction `json:"action"`
	RequestID    string           `json:"request_id"`
	Role         string           `json:"role"`
	Method       string           `json:"method"`
	Path         string           `json:"path"`
	Status       int              `json:"status"`
	PreviousHash string           `json:"previous_hash,omitempty"`
	Hash         string           `json:"hash"`
}

type statusCaptureWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusCaptureWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusCaptureWriter) Write(body []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(body)
}

func (s *ClassificationAPIServer) appendManagementAudit(
	route apiRoute,
	requestID string,
	principal managementPrincipal,
	request *http.Request,
	status int,
) {
	if s == nil || route.AuditAction == AuditActionNone {
		return
	}
	if status == 0 {
		status = http.StatusOK
	}
	s.managementAuditMu.Lock()
	defer s.managementAuditMu.Unlock()
	s.managementAuditSequence++
	entry := managementAuditEntry{
		Sequence:     s.managementAuditSequence,
		Timestamp:    time.Now().UTC().Format(time.RFC3339Nano),
		Action:       route.AuditAction,
		RequestID:    requestID,
		Role:         principal.Role,
		Method:       request.Method,
		Path:         route.Path,
		Status:       status,
		PreviousHash: s.managementAuditLastHash,
	}
	unsigned, _ := json.Marshal(entry)
	digest := sha256.Sum256(unsigned)
	entry.Hash = hex.EncodeToString(digest[:])
	// Keep a ring so retention does not copy thousands of entries per mutation.
	if len(s.managementAuditEntries) < maxManagementAuditEntries {
		s.managementAuditEntries = append(s.managementAuditEntries, entry)
	} else {
		s.managementAuditEntries[(entry.Sequence-1)%maxManagementAuditEntries] = entry
	}
	s.managementAuditLastHash = entry.Hash
}

type managementAuditResponse struct {
	Entries        []managementAuditEntry `json:"entries"`
	NextSequence   uint64                 `json:"next_sequence"`
	OldestSequence uint64                 `json:"oldest_sequence"`
	HasMore        bool                   `json:"has_more"`
	Truncated      bool                   `json:"truncated"`
	Retention      string                 `json:"retention"`
	Capacity       int                    `json:"capacity"`
}

func apiManagementAuditRoutes() []apiRoute {
	return []apiRoute{managedRoute(
		EndpointMetadata{Path: apiObservabilityPath + "/audit", Method: "GET", Description: "Page through this Router process's bounded management mutation audit; filter by action and resume after a sequence", Parameters: []OpenAPIParameter{
			queryParameter("action", "Exact mutation audit action to select.", "string"),
			queryParameter("after_sequence", "Exclusive sequence cursor; omitted starts at the oldest retained entry.", "integer"),
			queryParameter("limit", "Page size, from 1 to 1000; defaults to 100.", "integer"),
		}},
		routePolicy{Permission: PermAuditRead, Sensitivity: SensitivityOperational},
		(*ClassificationAPIServer).handleManagementAudit,
		jsonResponse[managementAuditResponse](http.StatusOK, "A bounded audit page; retention is process-local and resets on restart"),
		errorResponses(http.StatusBadRequest),
	)}
}

func (s *ClassificationAPIServer) handleManagementAudit(w http.ResponseWriter, r *http.Request) {
	var after uint64
	limit := 100
	var err error
	if raw := r.URL.Query().Get("after_sequence"); raw != "" {
		after, err = strconv.ParseUint(raw, 10, 64)
		if err != nil {
			s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_AUDIT_CURSOR", "after_sequence must be an unsigned integer")
			return
		}
	}
	if raw := r.URL.Query().Get("limit"); raw != "" {
		limit, err = strconv.Atoi(raw)
		if err != nil || limit < 1 || limit > 1000 {
			s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_AUDIT_LIMIT", "limit must be between 1 and 1000")
			return
		}
	}
	response := s.managementAuditPage(after, limit, RouteAuditAction(r.URL.Query().Get("action")))
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) managementAuditPage(after uint64, limit int, action RouteAuditAction) managementAuditResponse {
	s.managementAuditMu.Lock()
	defer s.managementAuditMu.Unlock()
	response := managementAuditResponse{Entries: []managementAuditEntry{}, NextSequence: after, Retention: "process", Capacity: maxManagementAuditEntries}
	if len(s.managementAuditEntries) > 0 {
		response.OldestSequence = s.managementAuditSequence - uint64(len(s.managementAuditEntries)) + 1
		response.Truncated = after > 0 && after < response.OldestSequence-1
	}
	for offset := range len(s.managementAuditEntries) {
		index := (response.OldestSequence - 1 + uint64(offset)) % maxManagementAuditEntries
		entry := s.managementAuditEntries[index]
		if entry.Sequence <= after {
			continue
		}
		if action != "" && entry.Action != action {
			response.NextSequence = entry.Sequence
			continue
		}
		if len(response.Entries) == limit {
			response.HasMore = true
			break
		}
		response.Entries = append(response.Entries, entry)
		response.NextSequence = entry.Sequence
	}
	return response
}
