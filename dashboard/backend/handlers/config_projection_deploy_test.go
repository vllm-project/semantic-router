package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

func setupTestConfigProjectionStore(t *testing.T) *configprojection.Store {
	t.Helper()

	previous := configProjectionStore
	store, err := configprojection.Open(filepath.Join(t.TempDir(), "projection.sqlite"))
	if err != nil {
		t.Fatalf("open config projection store: %v", err)
	}
	SetConfigProjectionStore(store)
	t.Cleanup(func() {
		SetConfigProjectionStore(previous)
		_ = store.Close()
	})
	return store
}

func TestDeployHandler_PersistsConfigProjectionAfterSuccessfulDeploy(t *testing.T) {
	store := setupTestConfigProjectionStore(t)
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	deployYAML := `routing:
  decisions:
    - name: projection-deploy-route
      priority: 9
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
`
	body := DeployRequest{
		YAML: deployYAML,
		DSL:  "route projection-deploy-route { model test-model }",
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal deploy request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	DeployHandler(configPath, false, tempDir)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("deploy status = %d, body = %s", w.Code, w.Body.String())
	}

	active, err := store.GetActiveProjection()
	if err != nil {
		t.Fatalf("GetActiveProjection: %v", err)
	}
	if active.Status != configprojection.StatusOK {
		t.Fatalf("expected active projection ok, got %+v", active)
	}
	if active.Deployment == nil {
		t.Fatal("expected active deployment record after deploy")
	}
	if active.Deployment.Source != configprojection.SourceDSL {
		t.Fatalf("expected source dsl, got %q", active.Deployment.Source)
	}
	if active.Deployment.DSLSnapshot != body.DSL {
		t.Fatalf("expected dsl snapshot %q, got %q", body.DSL, active.Deployment.DSLSnapshot)
	}

	deployments, err := store.ListDeployments()
	if err != nil {
		t.Fatalf("ListDeployments: %v", err)
	}
	if len(deployments) != 1 {
		t.Fatalf("expected 1 deployment record, got %+v", deployments)
	}
	if deployments[0].Source != configprojection.SourceDSL {
		t.Fatalf("expected listed deployment source dsl, got %q", deployments[0].Source)
	}
}

func TestRollbackHandler_RefreshesConfigProjectionAfterSuccessfulRollback(t *testing.T) {
	store := setupTestConfigProjectionStore(t)
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	deployYAML := `routing:
  decisions:
    - name: rollback-projection-route
      priority: 8
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: test-model
          use_reasoning: false
`
	deployBody := DeployRequest{YAML: deployYAML, DSL: "route rollback-projection-route {}"}
	deployBodyBytes, err := json.Marshal(deployBody)
	if err != nil {
		t.Fatalf("marshal deploy request: %v", err)
	}

	deployReq := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(deployBodyBytes))
	deployReq.Header.Set("Content-Type", "application/json")
	deployW := httptest.NewRecorder()
	DeployHandler(configPath, false, tempDir)(deployW, deployReq)
	if deployW.Code != http.StatusOK {
		t.Fatalf("deploy status = %d, body = %s", deployW.Code, deployW.Body.String())
	}

	versionsReq := httptest.NewRequest(http.MethodGet, "/api/router/config/versions", nil)
	versionsW := httptest.NewRecorder()
	ConfigVersionsHandler(configPath)(versionsW, versionsReq)
	if versionsW.Code != http.StatusOK {
		t.Fatalf("versions status = %d, body = %s", versionsW.Code, versionsW.Body.String())
	}

	var versions []ConfigVersion
	if decodeErr := json.NewDecoder(versionsW.Body).Decode(&versions); decodeErr != nil {
		t.Fatalf("decode versions: %v", decodeErr)
	}
	if len(versions) == 0 {
		t.Fatal("expected at least one backup version before rollback")
	}
	rollbackVersion := versions[0].Version

	rollbackBody, err := json.Marshal(map[string]string{"version": rollbackVersion})
	if err != nil {
		t.Fatalf("marshal rollback request: %v", err)
	}
	rollbackReq := httptest.NewRequest(http.MethodPost, "/api/router/config/rollback", bytes.NewReader(rollbackBody))
	rollbackReq.Header.Set("Content-Type", "application/json")
	rollbackW := httptest.NewRecorder()
	RollbackHandler(configPath, false, tempDir)(rollbackW, rollbackReq)
	if rollbackW.Code != http.StatusOK {
		t.Fatalf("rollback status = %d, body = %s", rollbackW.Code, rollbackW.Body.String())
	}

	active, err := store.GetActiveProjection()
	if err != nil {
		t.Fatalf("GetActiveProjection: %v", err)
	}
	if active.Status != configprojection.StatusOK {
		t.Fatalf("expected active projection ok after rollback, got %+v", active)
	}
	if active.ActiveVersion != rollbackVersion {
		t.Fatalf("expected active version %q after rollback, got %q", rollbackVersion, active.ActiveVersion)
	}

	rollbackDeployment, err := store.GetDeployment(rollbackVersion)
	if err != nil {
		t.Fatalf("GetDeployment(%q): %v", rollbackVersion, err)
	}
	if rollbackDeployment.Source != configprojection.SourceRollback {
		t.Fatalf("expected rollback deployment source, got %q", rollbackDeployment.Source)
	}

	backupData, err := os.ReadFile(filepath.Join(tempDir, ".vllm-sr", "config-backups", "config."+rollbackVersion+".yaml"))
	if err != nil {
		t.Fatalf("read rollback backup: %v", err)
	}
	if rollbackDeployment.YAMLHash == "" {
		t.Fatal("expected rollback deployment yaml hash")
	}
	if len(backupData) == 0 {
		t.Fatal("expected non-empty rollback backup content")
	}
}

func TestUpdateConfigHandler_PersistsConfigProjectionAfterSuccessfulUpdate(t *testing.T) {
	store := setupTestConfigProjectionStore(t)
	tempDir := t.TempDir()
	configPath := createLegacyTestConfig(t, tempDir)

	bodyBytes, err := json.Marshal(canonicalConfigBody("127.0.0.1:8000"))
	if err != nil {
		t.Fatalf("marshal update body: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, tempDir)(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("update status = %d, body = %s", w.Code, w.Body.String())
	}

	active, err := store.GetActiveProjection()
	if err != nil {
		t.Fatalf("GetActiveProjection: %v", err)
	}
	if active.Status != configprojection.StatusOK {
		t.Fatalf("expected active projection ok after update, got %+v", active)
	}
	if active.Deployment == nil {
		t.Fatal("expected active deployment record after update")
	}
	if active.Deployment.Source != configprojection.SourceManual {
		t.Fatalf("expected source manual, got %q", active.Deployment.Source)
	}

	deployments, err := store.ListDeployments()
	if err != nil {
		t.Fatalf("ListDeployments: %v", err)
	}
	if len(deployments) != 1 {
		t.Fatalf("expected 1 deployment record after update, got %+v", deployments)
	}
	if deployments[0].Source != configprojection.SourceManual {
		t.Fatalf("expected listed deployment source manual, got %q", deployments[0].Source)
	}
}
