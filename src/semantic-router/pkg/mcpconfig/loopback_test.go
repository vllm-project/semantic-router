package mcpconfig

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLoopbackOnly(t *testing.T) {
	t.Parallel()

	handler := LoopbackOnly(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	t.Run("allows loopback", func(t *testing.T) {
		t.Parallel()

		req := httptest.NewRequest(http.MethodGet, InternalHTTPPath, nil)
		req.RemoteAddr = "127.0.0.1:12345"
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		if rr.Code != http.StatusNoContent {
			t.Fatalf("status = %d, want %d", rr.Code, http.StatusNoContent)
		}
	})

	t.Run("blocks non-loopback", func(t *testing.T) {
		t.Parallel()

		req := httptest.NewRequest(http.MethodGet, InternalHTTPPath, nil)
		req.RemoteAddr = "203.0.113.10:12345"
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		if rr.Code != http.StatusForbidden {
			t.Fatalf("status = %d, want %d", rr.Code, http.StatusForbidden)
		}
	})
}
