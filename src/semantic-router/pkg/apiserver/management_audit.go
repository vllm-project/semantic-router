//go:build !windows

package apiserver

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
)

const maxManagementAuditEntries = 10000

type managementAuditEntry struct {
	Sequence  uint64           `json:"sequence"`
	Timestamp string           `json:"timestamp"`
	Action    RouteAuditAction `json:"action"`
	RequestID string           `json:"request_id"`
	Role      string           `json:"role"`
	Method    string           `json:"method"`
	Path      string           `json:"path"`
	Status    int              `json:"status"`
	// Config describes how a configuration update ended. Such an entry comes
	// from the Router's configuration lifecycle rather than from an HTTP
	// request, so its method, path and status are empty; its request ID and
	// role name the management request that caused it, if one did.
	Config       *configAuditEvent `json:"config,omitempty"`
	PreviousHash string            `json:"previous_hash,omitempty"`
	Hash         string            `json:"hash"`
}

// configAuditEvent records how one configuration update ended.
type configAuditEvent struct {
	Attempt uint64 `json:"attempt"`
	// Result is active (an ACK), failed (a NACK) or superseded.
	Result string `json:"result"`
	Source string `json:"source"`
	// Version is the configuration version an activation took.
	Version uint64 `json:"version,omitempty"`
	// Hash is the SHA-256 of the update's document.
	Hash string `json:"hash,omitempty"`
	// Stage and Codes say where and why a rejected update stopped.
	Stage      string   `json:"stage,omitempty"`
	Codes      []string `json:"codes,omitempty"`
	RollbackOf uint64   `json:"rollback_of,omitempty"`
}

var configAuditActions = map[configsnapshot.AttemptStatus]RouteAuditAction{
	configsnapshot.AttemptActive:     AuditActionConfigActivate,
	configsnapshot.AttemptFailed:     AuditActionConfigReject,
	configsnapshot.AttemptSuperseded: AuditActionConfigSupersede,
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
	s.appendAuditEntry(managementAuditEntry{
		Action:    route.AuditAction,
		RequestID: requestID,
		Role:      principal.Role,
		Method:    request.Method,
		Path:      route.Path,
		Status:    status,
	})
}

// recordConfigAudit appends a finished configuration update to the audit.
func (s *ClassificationAPIServer) recordConfigAudit(attempt configsnapshot.Attempt) {
	action, ok := configAuditActions[attempt.Status]
	if s == nil || !ok {
		return
	}
	event := &configAuditEvent{
		Attempt: attempt.ID, Result: string(attempt.Status), Source: string(attempt.Origin.Source),
		Version: attempt.Version, Hash: attempt.Hash, RollbackOf: attempt.Origin.RollbackOf,
	}
	if attempt.Status == configsnapshot.AttemptFailed {
		event.Stage = string(attempt.Stage)
		for _, reason := range attempt.Reasons {
			event.Codes = append(event.Codes, string(reason.Code))
		}
	}
	s.appendAuditEntry(managementAuditEntry{
		Action:    action,
		RequestID: attempt.Origin.RequestID,
		Role:      attempt.Origin.Principal,
		Config:    event,
	})
}

// appendAuditEntry chains entry to the audit.
func (s *ClassificationAPIServer) appendAuditEntry(entry managementAuditEntry) {
	s.managementAuditMu.Lock()
	defer s.managementAuditMu.Unlock()
	s.managementAuditSequence++
	entry.Sequence = s.managementAuditSequence
	entry.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	entry.PreviousHash = s.managementAuditLastHash
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
		EndpointMetadata{Path: apiObservabilityPath + "/audit", Method: "GET", Description: "Page through this Router process's bounded management mutation audit, configuration lifecycle outcomes included; filter by action and resume after a sequence", Parameters: []OpenAPIParameter{
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
