package router

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

func TestConfigDeployPermissionCannotMutateDatabaseOwnedEnterprisePolicy(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	store, err := auth.NewStore(filepath.Join(tempDir, "auth.db"))
	if err != nil {
		t.Fatalf("open auth store: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	service := auth.NewService(store, "enterprise-boundary-secret", 1)
	hash, err := service.HashPassword("ValidPassword1!")
	if err != nil {
		t.Fatalf("hash password: %v", err)
	}
	user, err := store.CreateUser(t.Context(), "config-deployer@example.com", "Config Deployer", hash, auth.RoleWrite, "active")
	if err != nil {
		t.Fatalf("create config deployer: %v", err)
	}
	token, _, err := service.Login(t.Context(), user.Email, "ValidPassword1!")
	if err != nil {
		t.Fatalf("login config deployer: %v", err)
	}

	beforePermissions, err := store.GetEffectivePermissions(t.Context(), user.Role, user.ID)
	if err != nil {
		t.Fatalf("load permissions before deploy: %v", err)
	}
	if !beforePermissions[auth.PermConfigDeploy] {
		t.Fatal("test principal must have config deploy permission")
	}
	for _, permission := range []string{
		auth.PermGrantPublish,
		auth.PermQuotaPublish,
		auth.PermVirtualKeys,
		auth.PermAuditPolicy,
		auth.PermBreakGlass,
	} {
		if beforePermissions[permission] {
			t.Fatalf("test principal unexpectedly has %q", permission)
		}
	}

	configPath := filepath.Join(tempDir, "config.yaml")
	originalConfig := []byte("version: v0.3\nrouting: {}\n")
	if err = os.WriteFile(configPath, originalConfig, 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	payload, err := json.Marshal(handlers.DeployRequest{YAML: `routing: {}
enterprise:
  tenant_grants: [{tenant: acme, role: admin}]
  quota_policy: {daily: 1000000}
  virtual_keys: [{name: hidden-key}]
  audit_policy: {enabled: false}
`})
	if err != nil {
		t.Fatalf("marshal deploy request: %v", err)
	}

	routes := auth.NewPolicyMux()
	routes.HandleFunc(
		auth.ProtectedMutationRoute(
			"/api/router/config/deploy",
			auth.PermConfigDeploy,
			"config.deploy",
			auth.SensitivitySecret,
			auth.ResourceOwnerConfig,
			16<<20,
			http.MethodPost,
		),
		handlers.DeployHandler(configPath, false, tempDir),
	)
	routes.Seal()
	handler := auth.AuthenticateRequest(service, routes)(routes)
	request := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy", bytes.NewReader(payload))
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)

	if response.Code != http.StatusForbidden {
		t.Fatalf("deploy status = %d, want %d: %s", response.Code, http.StatusForbidden, response.Body.String())
	}
	afterConfig, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config after deploy: %v", err)
	}
	if !bytes.Equal(afterConfig, originalConfig) {
		t.Fatalf("config changed after forbidden enterprise mutation:\n%s", afterConfig)
	}
	afterPermissions, err := store.GetEffectivePermissions(t.Context(), user.Role, user.ID)
	if err != nil {
		t.Fatalf("load permissions after deploy: %v", err)
	}
	if !reflect.DeepEqual(afterPermissions, beforePermissions) {
		t.Fatalf("effective permissions changed: before=%v after=%v", beforePermissions, afterPermissions)
	}
}
