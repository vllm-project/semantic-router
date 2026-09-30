//go:build !windows && cgo

package apiserver

import "net/http"

func (s *ClassificationAPIServer) handleStatus(w http.ResponseWriter, _ *http.Request) {
	if s.runtimeRegistry == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "RUNTIME_STATUS_UNAVAILABLE", "Replica-local runtime status is unavailable")
		return
	}
	s.writeJSONResponse(w, http.StatusOK, s.runtimeRegistry.Status())
}
