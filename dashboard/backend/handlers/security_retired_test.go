package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestRetiredSecurityPolicyHandlerAnswersGone pins the compatibility contract for the
// retired surface: every method, including the two that used to write router config,
// answers a stable 410 instead of being handled or proxied elsewhere.
func TestRetiredSecurityPolicyHandlerAnswersGone(t *testing.T) {
	t.Parallel()

	handler := RetiredSecurityPolicyHandler()

	for _, tc := range []struct {
		method string
		path   string
		body   string
	}{
		{method: http.MethodGet, path: "/api/security/policy"},
		{method: http.MethodPut, path: "/api/security/policy", body: `{"role_mappings":[],"rate_tiers":[]}`},
		{method: http.MethodPost, path: "/api/security/policy/preview", body: `{"role_mappings":[]}`},
		{method: http.MethodDelete, path: "/api/security/policy"},
	} {
		t.Run(tc.method+" "+tc.path, func(t *testing.T) {
			t.Parallel()

			req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			handler(rec, req)

			if rec.Code != http.StatusGone {
				t.Fatalf("status = %d, want %d", rec.Code, http.StatusGone)
			}
			if got := rec.Header().Get("Content-Type"); got != "application/json" {
				t.Fatalf("Content-Type = %q, want application/json", got)
			}

			var payload map[string]string
			if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
				t.Fatalf("response body is not JSON: %v (%q)", err, rec.Body.String())
			}
			if payload["error"] != "gone" {
				t.Fatalf("error = %q, want %q", payload["error"], "gone")
			}
			if payload["message"] == "" {
				t.Fatal("response carries no message explaining where the config now lives")
			}
		})
	}
}
