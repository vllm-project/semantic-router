package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// markerTTLSeconds tells the configured file's global block apart from the
// default config.yaml's.
const markerTTLSeconds = 4242

func TestRouterDefaultsHandlerUsesExactConfiguredPath(t *testing.T) {
	tempDir := t.TempDir()
	_ = createValidTestConfig(t, tempDir)

	customPath := filepath.Join(tempDir, "custom-global.yaml")
	customConfig := `
version: v0.5
global:
  services:
    response_api:
      enabled: true
      ttl_seconds: 4242
`
	if err := os.WriteFile(customPath, []byte(customConfig), 0o644); err != nil {
		t.Fatalf("write custom config: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/global", nil)
	recorder := httptest.NewRecorder()
	RouterDefaultsHandler(customPath)(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", recorder.Code, recorder.Body.String())
	}

	var response map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	responseAPI := requireMapValue(t, requireMapValue(t, response, "services"), "response_api")
	if got := responseAPI["ttl_seconds"]; got != float64(markerTTLSeconds) {
		t.Fatalf("response API ttl_seconds = %v, want %d from the configured file", got, markerTTLSeconds)
	}
}

func TestUpdateRouterDefaultsHandlerUsesExactConfiguredPath(t *testing.T) {
	tempDir := t.TempDir()
	decoyPath := createValidTestConfig(t, tempDir)
	decoyBefore, err := os.ReadFile(decoyPath)
	if err != nil {
		t.Fatalf("read decoy config: %v", err)
	}

	customPath := filepath.Join(tempDir, "custom-global.yaml")
	err = os.WriteFile(customPath, decoyBefore, 0o644)
	if err != nil {
		t.Fatalf("write custom config: %v", err)
	}

	patch := []byte(`{"services": {"response_api": {"enabled": true, "ttl_seconds": 4242}}}`)
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/global/update", bytes.NewReader(patch))
	recorder := httptest.NewRecorder()
	UpdateRouterDefaultsHandler(customPath, false, tempDir)(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", recorder.Code, recorder.Body.String())
	}
	decoyAfter, err := os.ReadFile(decoyPath)
	if err != nil {
		t.Fatalf("read decoy config after update: %v", err)
	}
	if !bytes.Equal(decoyAfter, decoyBefore) {
		t.Fatal("default config.yaml changed instead of the configured custom file")
	}

	updatedGlobal, err := currentGlobalDefaults(customPath)
	if err != nil {
		t.Fatalf("read updated custom config: %v", err)
	}
	if got := updatedGlobal.Services.ResponseAPI.TTLSeconds; got != markerTTLSeconds {
		t.Fatalf("updated response API ttl_seconds = %d, want %d", got, markerTTLSeconds)
	}
}

func requireMapValue(t *testing.T, value map[string]any, key string) map[string]any {
	t.Helper()
	nested, ok := value[key].(map[string]any)
	if !ok {
		t.Fatalf("%s = %#v, want object", key, value[key])
	}
	return nested
}
