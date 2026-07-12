package handlers

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStaticFileServerServesLoginAsSPARoute(t *testing.T) {
	staticDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("<html>app</html>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/login", nil)

	StaticFileServer(staticDir).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), "app") {
		t.Fatalf("expected SPA index body, got %q", recorder.Body.String())
	}
}

func TestStaticFileServerKeepsProxyRoutesReserved(t *testing.T) {
	staticDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("<html>app</html>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/public/build/grafana.js", nil)

	StaticFileServer(staticDir).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("expected status 502, got %d", recorder.Code)
	}
}

func TestStaticFileServerProvidesWellKnownChangePasswordRedirect(t *testing.T) {
	t.Parallel()

	staticDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("<html>app</html>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/.well-known/change-password", nil)
	StaticFileServer(staticDir).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusFound {
		t.Fatalf("status = %d, want 302", recorder.Code)
	}
	if location := recorder.Header().Get("Location"); location != "/account/security" {
		t.Fatalf("Location = %q, want /account/security", location)
	}
}

func TestStaticFileServerReturnsReal404ForUnknownWellKnownResources(t *testing.T) {
	t.Parallel()

	staticDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("<html>app</html>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}

	paths := []string{
		"/.well-known/resource-that-should-not-exist-whose-status-code-should-not-be-200",
		"/.well-known/unknown-resource",
	}
	for _, requestPath := range paths {
		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, requestPath, nil)
		StaticFileServer(staticDir).ServeHTTP(recorder, req)
		if recorder.Code != http.StatusNotFound {
			t.Fatalf("%s status = %d, want 404", requestPath, recorder.Code)
		}
		if strings.Contains(recorder.Body.String(), "<html>app</html>") {
			t.Fatalf("%s unexpectedly received SPA shell", requestPath)
		}
	}
}

func TestStaticFileServerKeepsAccountSecurityAsSPARoute(t *testing.T) {
	t.Parallel()

	staticDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("<html>app</html>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/account/security", nil)
	StaticFileServer(staticDir).ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), "app") {
		t.Fatalf("status/body = %d %q, want SPA route", recorder.Code, recorder.Body.String())
	}
}
