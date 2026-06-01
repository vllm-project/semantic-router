package router

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// When the auth service fails to initialize (authSvc == nil), wrapWithAuth must
// fail CLOSED: routes that require authentication must be denied with 503 and
// their backend handlers must never run, while public/static routes stay
// reachable so the dashboard can surface the misconfiguration. This guards
// against the fail-open regression where the entire control plane was served
// unauthenticated whenever the auth store could not be opened.
func TestWrapWithAuthFailsClosedWhenAuthUnavailable(t *testing.T) {
	t.Parallel()

	protectedHit := false
	publicHit := false

	mux := http.NewServeMux()
	mux.HandleFunc("/api/router/config", func(w http.ResponseWriter, _ *http.Request) {
		protectedHit = true
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		publicHit = true
		w.WriteHeader(http.StatusOK)
	})

	handler := wrapWithAuth(mux, nil) // nil => auth store failed to initialize

	t.Run("protected route denied without executing handler", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/router/config", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("protected route status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
		}
		if protectedHit {
			t.Fatal("protected handler executed while auth was unavailable (fail-open regression)")
		}
	})

	t.Run("public/static route still served", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("public route status = %d, want %d", rec.Code, http.StatusOK)
		}
		if !publicHit {
			t.Fatal("public handler did not run; fail-closed guard over-blocked")
		}
	})

	t.Run("protected admin route denied", func(t *testing.T) {
		mux.HandleFunc("/api/admin/users", func(w http.ResponseWriter, _ *http.Request) {
			t.Error("admin handler executed while auth was unavailable")
			w.WriteHeader(http.StatusOK)
		})
		req := httptest.NewRequest(http.MethodGet, "/api/admin/users", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("admin route status = %d, want %d", rec.Code, http.StatusServiceUnavailable)
		}
	})
}
