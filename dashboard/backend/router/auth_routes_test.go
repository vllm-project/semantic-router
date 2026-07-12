package router

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
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

func TestSetupAuthRoutesFailsClosedForInvalidPasswordBlocklist(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	cfg := &config.Config{
		AuthDBPath:            filepath.Join(t.TempDir(), "auth.db"),
		PasswordBlocklistPath: filepath.Join(t.TempDir(), "missing-password-blocklist.txt"),
	}
	svc, err := setupAuthRoutes(mux, cfg)
	if svc != nil {
		t.Fatal("setupAuthRoutes() returned a service for an invalid blocklist")
	}
	if err == nil {
		t.Fatal("setupAuthRoutes() accepted an invalid blocklist")
	}
	server, setupErr := Setup(cfg)
	if server != nil || setupErr == nil {
		t.Fatalf("Setup() = (%#v, %v), want blocklist startup failure", server, setupErr)
	}
}

func TestSetupAuthRoutesFailsClosedForWeakConfiguredJWTSecret(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	cfg := &config.Config{
		AuthDBPath: filepath.Join(t.TempDir(), "auth.db"),
		JWTSecret:  "too-short",
	}
	svc, err := setupAuthRoutes(mux, cfg)
	if svc != nil {
		t.Fatal("setupAuthRoutes() returned a service for a weak JWT secret")
	}
	if err == nil {
		t.Fatal("setupAuthRoutes() accepted a weak JWT secret")
	}
	if strings.Contains(err.Error(), cfg.JWTSecret) {
		t.Fatalf("setup error exposed configured JWT secret: %q", err)
	}

	server, setupErr := Setup(cfg)
	if server != nil || setupErr == nil {
		t.Fatalf("Setup() = (%#v, %v), want startup failure", server, setupErr)
	}
}
