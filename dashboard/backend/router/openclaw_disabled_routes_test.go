package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// With the feature at its default, every OpenClaw collection route answers
// the disabled contract the item route and the embedded view already use.
// An empty array would be indistinguishable from a disabled feature with no
// data, and it invites the user to run operations the disabled branch does
// not register (the GET-only registration is enforced by the route contract,
// which is not what this test exercises).
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
