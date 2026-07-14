package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHandleCORSPreflightRequiresSameOrigin(t *testing.T) {
	for _, test := range []struct {
		name       string
		origin     string
		wantStatus int
		wantOrigin string
	}{
		{name: "same origin", origin: "https://dashboard.example.com", wantStatus: http.StatusNoContent, wantOrigin: "https://dashboard.example.com"},
		{name: "same-site sibling origin", origin: "https://evil.example.com", wantStatus: http.StatusForbidden, wantOrigin: ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodOptions, "http://dashboard.example.com/api/status", nil)
			req.Header.Set("Origin", test.origin)
			req.Header.Set("X-Forwarded-Proto", "https")
			req.Header.Set("Access-Control-Request-Private-Network", "true")
			recorder := httptest.NewRecorder()

			if handled := HandleCORSPreflight(recorder, req); !handled {
				t.Fatal("preflight was not handled")
			}
			if recorder.Code != test.wantStatus {
				t.Fatalf("status = %d, want %d", recorder.Code, test.wantStatus)
			}
			if got := recorder.Header().Get("Access-Control-Allow-Origin"); got != test.wantOrigin {
				t.Fatalf("Access-Control-Allow-Origin = %q, want %q", got, test.wantOrigin)
			}
			wantPrivateNetwork := ""
			if test.wantOrigin != "" {
				wantPrivateNetwork = "true"
			}
			if got := recorder.Header().Get("Access-Control-Allow-Private-Network"); got != wantPrivateNetwork {
				t.Fatalf("Access-Control-Allow-Private-Network = %q, want %q", got, wantPrivateNetwork)
			}
		})
	}
}
