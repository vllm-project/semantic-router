//go:build !windows && cgo

package apiserver

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"time"
)

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
	s.managementAuditEntries = append(s.managementAuditEntries, entry)
	s.managementAuditLastHash = entry.Hash
}

func (s *ClassificationAPIServer) responseCacheAuditEntries() []managementAuditEntry {
	if s == nil {
		return nil
	}
	s.managementAuditMu.Lock()
	defer s.managementAuditMu.Unlock()
	entries := make([]managementAuditEntry, 0, len(s.managementAuditEntries))
	for _, entry := range s.managementAuditEntries {
		if entry.Action == AuditActionCacheInvalidate || entry.Action == AuditActionCacheFlush {
			entries = append(entries, entry)
		}
	}
	return entries
}

func (s *ClassificationAPIServer) contextCompressionAuditEntries() []managementAuditEntry {
	if s == nil {
		return nil
	}
	s.managementAuditMu.Lock()
	defer s.managementAuditMu.Unlock()
	entries := make([]managementAuditEntry, 0, len(s.managementAuditEntries))
	for _, entry := range s.managementAuditEntries {
		if entry.Action == AuditActionCompressionPreview ||
			entry.Action == AuditActionCompressionInvalidate {
			entries = append(entries, entry)
		}
	}
	return entries
}

func (s *ClassificationAPIServer) handleResponseCacheAudit(
	w http.ResponseWriter,
	_ *http.Request,
) {
	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"entries": s.responseCacheAuditEntries(),
	})
}
