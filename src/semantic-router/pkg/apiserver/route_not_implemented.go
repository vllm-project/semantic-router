//go:build !windows && cgo

package apiserver

import (
	"net/http"
)

func (s *ClassificationAPIServer) handleClassificationMetrics(w http.ResponseWriter, _ *http.Request) {
	s.writeErrorResponse(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Classification metrics not implemented yet")
}

func (s *ClassificationAPIServer) handleGetConfig(w http.ResponseWriter, _ *http.Request) {
	s.writeErrorResponse(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Get config not implemented yet")
}

func (s *ClassificationAPIServer) handleUpdateConfig(w http.ResponseWriter, _ *http.Request) {
	s.writeErrorResponse(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Update config not implemented yet")
}

// Placeholder handlers for remaining endpoints
func (s *ClassificationAPIServer) handleCombinedClassification(w http.ResponseWriter, _ *http.Request) {
	s.writeErrorResponse(w, http.StatusNotImplemented, "NOT_IMPLEMENTED", "Combined classification not implemented yet")
}
