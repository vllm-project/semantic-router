package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Every collection route answers the disabled contract the item route already uses.
func TestDisabledOpenClawRoutesReportFeatureDisabled(t *testing.T) {
	var mux http.ServeMux
	registerDisabledOpenClawRoutes(&mux)
	for _, path := range []string{"/api/openclaw/status", "/api/openclaw/teams", "/api/openclaw/workers", "/api/openclaw/rooms", "/api/openclaw/rooms/room-1"} {
		recorder := httptest.NewRecorder()
		mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, nil))
		if recorder.Code != http.StatusServiceUnavailable {
			t.Fatalf("GET %s: expected 503, got %d", path, recorder.Code)
		}
		if body := recorder.Body.String(); !strings.Contains(body, `"error":"OpenClaw feature disabled"`) {
			t.Fatalf("GET %s: expected the disabled error body, got %s", path, body)
		}
	}
}
