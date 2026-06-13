package handlers

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

func TestReadCanonicalConfigPreferProjectionUsesActiveSnapshot(t *testing.T) {
	store := setupTestConfigProjectionStore(t)
	SetConfigProjectionStore(store)

	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  decisions: []\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	canonicalYAML := testCanonicalYAMLForProjection()
	if err := store.RefreshFromCanonical(configprojection.RefreshInput{
		Version:   "20260101-120000",
		Source:    configprojection.SourceManual,
		YAMLBytes: []byte(canonicalYAML),
	}); err != nil {
		t.Fatalf("RefreshFromCanonical: %v", err)
	}

	cfg, source, err := readCanonicalConfigPreferProjection(configPath)
	if err != nil {
		t.Fatalf("readCanonicalConfigPreferProjection: %v", err)
	}
	if source != "projection" {
		t.Fatalf("source = %q, want projection", source)
	}
	if cfg == nil || len(cfg.Routing.Decisions) == 0 {
		t.Fatalf("expected projected decisions, got %+v", cfg)
	}
}

func TestReadConfigYAMLPreferProjectionFallsBackToFile(t *testing.T) {
	previous := configProjectionStore
	SetConfigProjectionStore(nil)
	t.Cleanup(func() {
		SetConfigProjectionStore(previous)
	})

	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	want := "version: v0.3\nrouting:\n  decisions: []\n"
	if err := os.WriteFile(configPath, []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}

	got, source, err := readConfigYAMLPreferProjection(configPath)
	if err != nil {
		t.Fatalf("readConfigYAMLPreferProjection: %v", err)
	}
	if source != "file" {
		t.Fatalf("source = %q, want file", source)
	}
	if string(got) != want {
		t.Fatalf("yaml = %q, want %q", string(got), want)
	}
}

func TestConfigHandlerServesProjectionWithHeader(t *testing.T) {
	store := setupTestConfigProjectionStore(t)
	SetConfigProjectionStore(store)

	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  decisions: []\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.RefreshFromCanonical(configprojection.RefreshInput{
		Version:   "20260101-120000",
		Source:    configprojection.SourceManual,
		YAMLBytes: []byte(testCanonicalYAMLForProjection()),
	}); err != nil {
		t.Fatalf("RefreshFromCanonical: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil)
	w := httptest.NewRecorder()
	ConfigHandler(configPath)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("X-Config-Source"); got != "projection" {
		t.Fatalf("X-Config-Source = %q, want projection", got)
	}
}

func TestConfigVersionsHandlerPrefersProjectionDeployments(t *testing.T) {
	store := setupTestConfigProjectionStore(t)
	SetConfigProjectionStore(store)

	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(testCanonicalYAMLForProjection()), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.RefreshFromCanonical(configprojection.RefreshInput{
		Version:   "20260101-120000",
		Source:    configprojection.SourceDSL,
		YAMLBytes: []byte(testCanonicalYAMLForProjection()),
	}); err != nil {
		t.Fatalf("RefreshFromCanonical: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/versions", nil)
	w := httptest.NewRecorder()
	ConfigVersionsHandler(configPath)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("X-Config-Versions-Source"); got != "projection" {
		t.Fatalf("X-Config-Versions-Source = %q, want projection", got)
	}
}
