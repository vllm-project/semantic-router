package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestConfigVersionsHandlerUsesConfiguredStateDirectory(t *testing.T) {
	root := t.TempDir()
	configPath := filepath.Join(root, "mounted", "config.yaml")
	stateDir := filepath.Join(root, "state")
	t.Setenv("DASHBOARD_CONFIG_DIR", stateDir)
	backupDir := configBackupDir(stateDir)
	if err := os.MkdirAll(backupDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(backupDir, "config.20260924-120000.yaml"), []byte("version: v0.4\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	response := httptest.NewRecorder()
	ConfigVersionsHandler(configPath)(response, httptest.NewRequest(http.MethodGet, "/api/router/config/versions", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	var versions []ConfigVersion
	if err := json.Unmarshal(response.Body.Bytes(), &versions); err != nil {
		t.Fatal(err)
	}
	if len(versions) != 1 || versions[0].Version != "20260924-120000" {
		t.Fatalf("versions = %+v", versions)
	}
}
