package handlers

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWizMapStaticHandlerServesIndexAndAssets(t *testing.T) {
	t.Parallel()

	staticDir := t.TempDir()
	root := filepath.Join(staticDir, "embedded", "wizmap")
	if err := os.MkdirAll(filepath.Join(root, "assets"), 0o755); err != nil {
		t.Fatalf("mkdir assets: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "index.html"), []byte("<html>knowledge map</html>"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "assets", "app.js"), []byte("console.log('wizmap');"), 0o644); err != nil {
		t.Fatalf("write asset: %v", err)
	}

	handler := WizMapStaticHandler(staticDir)

	indexReq := httptest.NewRequest(http.MethodGet, "/embedded/wizmap/", nil)
	indexRR := httptest.NewRecorder()
	handler(indexRR, indexReq)

	if indexRR.Code != http.StatusOK {
		t.Fatalf("expected 200 for index, got %d", indexRR.Code)
	}
	if !strings.Contains(indexRR.Body.String(), "knowledge map") {
		t.Fatalf("expected index body, got %q", indexRR.Body.String())
	}

	assetReq := httptest.NewRequest(http.MethodGet, "/embedded/wizmap/assets/app.js", nil)
	assetRR := httptest.NewRecorder()
	handler(assetRR, assetReq)

	if assetRR.Code != http.StatusOK {
		t.Fatalf("expected 200 for asset, got %d", assetRR.Code)
	}
	if !strings.Contains(assetRR.Body.String(), "wizmap") {
		t.Fatalf("expected asset body, got %q", assetRR.Body.String())
	}
}

func TestWizMapStaticHandlerReturnsUnavailableWhenBundleMissing(t *testing.T) {
	t.Parallel()

	handler := WizMapStaticHandler(t.TempDir())
	req := httptest.NewRequest(http.MethodGet, "/embedded/wizmap/", nil)
	rr := httptest.NewRecorder()
	handler(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "Knowledge Map is not available yet") {
		t.Fatalf("expected unavailable message, got %q", rr.Body.String())
	}
}
