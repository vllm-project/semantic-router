package router

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestReadRoleCannotObtainListenerAPIKeys(t *testing.T) {
	const listenerKey = "listener-client-key-fixture"
	dir := t.TempDir()
	cfg := &config.Config{
		AbsConfigPath: filepath.Join(dir, "config.yaml"), ConfigDir: dir,
		AuthDBPath: filepath.Join(dir, "auth.db"), JWTSecret: "listener-test-signing-key", JWTExpiryHours: 1,
		RuntimeConfigWritable: true, RouterAPIURL: "http://127.0.0.1:1",
	}
	keyed := "version: v0.5\nlisteners:\n  - name: http-8899\n    address: 0.0.0.0\n    port: 8899\n    api_keys:\n      - " + listenerKey + "\n"
	if err := os.WriteFile(cfg.AbsConfigPath, []byte(keyed), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := auth.NewStore(cfg.AuthDBPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	service := auth.NewService(store, cfg.JWTSecret, cfg.JWTExpiryHours)
	mux := auth.NewPolicyMux()
	registerConfigRoutes(mux, cfg)
	registerRouterAPIProxy(mux, cfg, nil, nil, nil, nil)
	mux.Seal()
	handler := wrapWithAuth(mux, service, mux)
	session := func(email, role string) string {
		t.Helper()
		const password = "listener-api-keys-test"
		hash, err := service.HashPassword(password)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = store.CreateUser(t.Context(), email, email, hash, role, "active"); err != nil {
			t.Fatal(err)
		}
		token, _, err := service.Login(t.Context(), email, password)
		if err != nil {
			t.Fatal(err)
		}
		return token
	}
	get := func(token, path string) *httptest.ResponseRecorder {
		t.Helper()
		request := httptest.NewRequest(http.MethodGet, path, nil)
		request.Header.Set("Authorization", "Bearer "+token)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}

	reader := session("reader@example.test", auth.RoleRead)
	writer := session("writer@example.test", auth.RoleWrite)
	for _, path := range []string{"/api/router/config/all", "/api/router/config/yaml"} {
		response := get(reader, path)
		if response.Code != http.StatusOK || strings.Contains(response.Body.String(), listenerKey) || !strings.Contains(response.Body.String(), "8899") {
			t.Fatalf("read role %s: status=%d body=%s", path, response.Code, response.Body.String())
		}
		response = get(writer, path)
		if response.Code != http.StatusOK || strings.Contains(response.Body.String(), listenerKey) {
			t.Fatalf("write role %s: status=%d body=%s", path, response.Code, response.Body.String())
		}
	}
	if response := get(reader, "/api/router/api/v1/inventory/classifier"); response.Code != http.StatusForbidden {
		t.Fatalf("read role classifier inventory status=%d", response.Code)
	}
}
