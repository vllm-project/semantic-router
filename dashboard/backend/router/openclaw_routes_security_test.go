package router

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProductionOpenClawEmbeddedUIFailsClosedWithoutOriginIsolation(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	registerProductionOpenClawProxyBoundary(mux)
	request := httptest.NewRequest(
		http.MethodGet,
		"https://dashboard.example/embedded/openclaw/worker-a/",
		nil,
	)
	recorder := httptest.NewRecorder()

	mux.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
	}
}
