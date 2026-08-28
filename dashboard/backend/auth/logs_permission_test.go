package auth

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRuntimeLogsAreDeniedToDefaultReadRole(t *testing.T) {
	t.Parallel()

	service := newTestAuthService(t)
	reader := newTestUser(t, service, "logs-reader@example.com", RoleRead, "active")
	writer := newTestUser(t, service, "logs-writer@example.com", RoleWrite, "active")
	var calls int
	handler := protectedTestHandler(service, ProtectedRoute("/api/logs", PermLogsRead, SensitivitySensitive, ResourceOwnerObservability, http.MethodGet), func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusNoContent)
	})

	readerResponse := httptest.NewRecorder()
	handler.ServeHTTP(
		readerResponse,
		newAuthenticatedRequest(t, service, reader, http.MethodGet, "/api/logs", ""),
	)
	if readerResponse.Code != http.StatusForbidden || calls != 0 {
		t.Fatalf("reader status = %d, calls = %d", readerResponse.Code, calls)
	}

	writerResponse := httptest.NewRecorder()
	handler.ServeHTTP(
		writerResponse,
		newAuthenticatedRequest(t, service, writer, http.MethodGet, "/api/logs", ""),
	)
	if writerResponse.Code != http.StatusNoContent || calls != 1 {
		t.Fatalf("writer status = %d, calls = %d", writerResponse.Code, calls)
	}
}
