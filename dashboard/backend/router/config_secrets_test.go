package router

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

func TestConfigSecretsThroughAuthentication(t *testing.T) {
	const secret = "dashboard-auth-config-secret-fixture"
	t.Setenv("CONFIG_AUTH_TEST_SECRET", secret)
	for _, source := range []string{secret, "${CONFIG_AUTH_TEST_SECRET}"} {
		t.Run(source, func(t *testing.T) {
			dir := t.TempDir()
			configPath := filepath.Join(dir, "config.yaml")
			fixture := "version: v0.3\nglobal:\n  services:\n    response_api:\n      enabled: true\n      store_backend: redis\n      redis:\n        address: 127.0.0.1:6379\n        password: " + source + "\n"
			if err := os.WriteFile(configPath, []byte(fixture), 0o600); err != nil {
				t.Fatal(err)
			}
			store, err := auth.NewStore(filepath.Join(dir, "auth.sqlite"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = store.Close() })
			svc := auth.NewService(store, "test-only-config-signing-key", 1)
			cfg := &config.Config{AbsConfigPath: configPath, ConfigDir: dir, ReadonlyMode: true, RouterAPIURL: "http://127.0.0.1:1"}
			mux := auth.NewPolicyMux()
			registerAuthProxyRoutes(mux, svc)
			registerConfigRoutes(mux, cfg, configRouteOptions{credentialStore: recipe.NewStore(recipe.StoreOptions{Root: filepath.Join(dir, "recipes"), ConfigPath: configPath})})
			mux.Seal()
			handler := wrapWithAuth(mux, svc, mux)
			db, err := sql.Open("sqlite3", filepath.Join(dir, "auth.sqlite"))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = db.Close() })
			const password = "test-only-config-login-password"
			hash, err := svc.HashPassword(password)
			if err != nil {
				t.Fatal(err)
			}
			type session struct{ token, userID string }
			login := func(role string) session {
				t.Helper()
				user, err := store.CreateUser(t.Context(), role+"@example.invalid", role, hash, role, "active")
				if err != nil {
					t.Fatal(err)
				}
				body, _ := json.Marshal(auth.LoginRequest{Email: user.Email, Password: password})
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/auth/login", bytes.NewReader(body)))
				var result auth.LoginResponse
				if response.Code != http.StatusOK {
					t.Fatalf("login status = %d", response.Code)
				}
				if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
					t.Fatal(err)
				}
				return session{token: result.Token, userID: user.ID}
			}
			get := func(path, token string) *httptest.ResponseRecorder {
				request := httptest.NewRequest(http.MethodGet, path, nil)
				if token != "" {
					request.Header.Set("Authorization", "Bearer "+token)
				}
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, request)
				return response
			}
			reader, writer, admin := login(auth.RoleRead), login(auth.RoleWrite), login(auth.RoleAdmin)
			for _, path := range []string{"/api/router/config/all", "/api/router/config/yaml", "/api/router/config/global", "/api/router/config/global/raw"} {
				for _, actor := range []struct {
					name, token string
					status      int
					visible     bool
				}{
					{name: "anonymous", status: 401},
					{name: "invalid token", token: "invalid", status: 401},
					{name: "reader", token: reader.token, status: 200},
					{name: "writer", token: writer.token, status: 200},
					{name: "admin", token: admin.token, status: 200, visible: source == secret},
				} {
					response := get(path, actor.token)
					if response.Code != actor.status || strings.Contains(response.Body.String(), secret) != actor.visible {
						t.Errorf("%s %s: status = %d, plaintext visible = %v", actor.name, path, response.Code, strings.Contains(response.Body.String(), secret))
					}
				}
			}
			// A live permission change must affect the existing JWT session.
			if _, err := db.Exec("INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES (?, ?, 0)", admin.userID, auth.PermConfigSecretsRead); err != nil {
				t.Fatal(err)
			}
			if result := get("/api/router/config/global/raw", admin.token); result.Code != 200 || strings.Contains(result.Body.String(), secret) {
				t.Fatal("revoked secret-view permission still exposed credentials")
			}
			if _, err := db.Exec("INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES (?, ?, 0)", reader.userID, auth.PermConfigRead); err != nil {
				t.Fatal(err)
			}
			if result := get("/api/router/config/global/raw", reader.token); result.Code != 403 || strings.Contains(result.Body.String(), secret) {
				t.Fatal("revoked config-read permission did not fail closed")
			}
		})
	}
}
