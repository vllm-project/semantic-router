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

func TestStaticFileServerFailsProbePathsLoudly(t *testing.T) {
	staticDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("<html>app</html>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}

	for _, path := range []string{"/health", "/health/", "/ready", "/live"} {
		recorder := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, path, nil)

		StaticFileServer(staticDir).ServeHTTP(recorder, req)

		if recorder.Code != http.StatusNotFound {
			t.Fatalf("probe %s: status=%d, want 404 instead of the SPA shell", path, recorder.Code)
		}
		if strings.Contains(recorder.Body.String(), "app") {
			t.Fatalf("probe %s: body is the SPA shell", path)
		}
	}

	// A real SPA route still takes the shell: the guard covers probe paths only.
	recorder := httptest.NewRecorder()
	StaticFileServer(staticDir).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/login", nil))
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), "app") {
		t.Fatalf("SPA route: status=%d body=%q", recorder.Code, recorder.Body.String())
	}
}
