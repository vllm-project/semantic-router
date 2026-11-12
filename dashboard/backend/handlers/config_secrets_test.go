package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
	sigsyaml "sigs.k8s.io/yaml"
)

const (
	configSecretFixture    = "dashboard-config-secret-fixture"
	configSecretPermission = "config.secrets.read"
)

func writeSecretConfig(t *testing.T, password string) string {
	t.Helper()
	path := createValidTestConfig(t, t.TempDir())
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, []byte("\nglobal:\n  services:\n    response_api:\n      enabled: true\n      store_backend: redis\n      redis:\n        address: 127.0.0.1:6379\n        password: "+password+"\n")...)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestConfigReadEndpointsRedactServiceSecrets(t *testing.T) {
	t.Setenv("DASHBOARD_CONFIG_TEST_PASSWORD", configSecretFixture)
	for _, source := range []string{configSecretFixture, "${DASHBOARD_CONFIG_TEST_PASSWORD}"} {
		path := writeSecretConfig(t, source)
		for _, actor := range []struct {
			name  string
			perms []string
		}{
			{name: "no principal"},
			{name: "reader", perms: []string{"config.read"}},
			{name: "writer", perms: []string{"config.read", "config.write"}},
			{name: "deployer", perms: []string{"config.read", "config.deploy"}},
			{name: "secret viewer", perms: []string{"config.read", configSecretPermission}},
		} {
			for _, endpoint := range []struct {
				path    string
				handler http.HandlerFunc
			}{
				{"/api/router/config/all", ConfigHandler(path, false)},
				{"/api/router/config/yaml", ConfigYAMLHandler(path, false)},
				{"/api/router/config/global", RouterDefaultsHandler(path)},
				{"/api/router/config/global/raw", GlobalConfigYAMLHandler(path)},
			} {
				t.Run(actor.name+"/"+endpoint.path+"/"+source, func(t *testing.T) {
					response := httptest.NewRecorder()
					endpoint.handler(response, requestAs(endpoint.path, actor.perms...))
					if response.Code != http.StatusOK {
						t.Fatalf("status = %d: %s", response.Code, response.Body.String())
					}
					body := response.Body.String()
					wantVisible := actor.name == "secret viewer" && source == configSecretFixture
					if visible := strings.Contains(body, configSecretFixture); visible != wantVisible {
						t.Errorf("plaintext secret visible = %v, want %v", visible, wantVisible)
					}
					if source != configSecretFixture && !strings.Contains(body, source) {
						t.Error("environment reference was not preserved")
					}
					if !strings.Contains(response.Header().Get("Cache-Control"), "no-store") {
						t.Error("configuration response can be cached")
					}
				})
			}
		}
	}
}

func TestGlobalConfigEditorPreservesRedactedSecret(t *testing.T) {
	for _, raw := range []bool{false, true} {
		t.Run(map[bool]string{false: "json", true: "yaml"}[raw], func(t *testing.T) {
			path := writeSecretConfig(t, configSecretFixture)
			response := httptest.NewRecorder()
			var read, update http.HandlerFunc
			if raw {
				read = GlobalConfigYAMLHandler(path)
				update = UpdateGlobalConfigYAMLHandler(path, false, filepath.Dir(path))
			} else {
				read = RouterDefaultsHandler(path)
				update = UpdateRouterDefaultsHandler(path, false, filepath.Dir(path))
			}
			read(response, requestAs("/", "config.read", "config.write"))
			if response.Code != http.StatusOK {
				t.Fatalf("read status = %d", response.Code)
			}
			if strings.Contains(response.Body.String(), configSecretFixture) {
				t.Fatal("editor received plaintext secret")
			}
			previousPropagate := propagateConfig
			propagateConfig = func(string, string) error { return nil }
			t.Cleanup(func() { propagateConfig = previousPropagate })
			request := httptest.NewRequest(http.MethodPut, "/", bytes.NewReader(response.Body.Bytes()))
			request = request.WithContext(requestAs("/", "config.write").Context())
			result := httptest.NewRecorder()
			update(result, request)
			if result.Code != http.StatusOK {
				t.Fatalf("save status = %d: %s", result.Code, result.Body.String())
			}
			saved, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(saved, []byte(configSecretFixture)) {
				t.Fatal("saving the redacted view lost the stored secret")
			}
		})
	}
}

func TestGlobalConfigEditorRejectsRedisClusterRetargeting(t *testing.T) {
	for _, raw := range []bool{false, true} {
		for _, tc := range []struct {
			name, patch string
			standalone  bool
		}{
			{name: "replace node", patch: "cluster_addresses: [different.example.test:6379]"},
			{name: "add node", patch: "cluster_addresses: [redis-a.example.test:6379, redis-b.example.test:6379, different.example.test:6379]"},
			{name: "remove node", patch: "cluster_addresses: [redis-a.example.test:6379]"},
			{name: "remove nodes field", patch: "cluster_addresses: null"},
			{name: "disable cluster", patch: "cluster_mode: false"},
			{name: "introduce cluster", standalone: true, patch: "cluster_mode: true\ncluster_addresses: [different.example.test:6379]"},
		} {
			t.Run(map[bool]string{false: "json", true: "yaml"}[raw]+"/"+tc.name, func(t *testing.T) {
				path := writeSecretConfig(t, configSecretFixture)
				original, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				if !tc.standalone {
					original = append(original, []byte("        cluster_mode: true\n        cluster_addresses: [redis-a.example.test:6379, redis-b.example.test:6379]\n")...)
				}
				if writeErr := os.WriteFile(path, original, 0o600); writeErr != nil {
					t.Fatal(writeErr)
				}
				read, update := globalSecretEditorHandlers(path, raw)
				view := httptest.NewRecorder()
				read(view, requestAs("/", "config.read", "config.write"))
				if view.Code != http.StatusOK || strings.Contains(view.Body.String(), configSecretFixture) {
					t.Fatalf("redacted read status=%d", view.Code)
				}
				document, err := configSecretDocument(view.Body.Bytes())
				if err != nil {
					t.Fatal(err)
				}
				redis := mappingValueNode(mappingValueNode(mappingValueNode(document, "services"), "response_api"), "redis")
				patch, err := configSecretDocument([]byte(tc.patch))
				if err != nil {
					t.Fatal(err)
				}
				for i := 0; i+1 < len(patch.Content); i += 2 {
					setMappingValueNode(redis, patch.Content[i].Value, patch.Content[i+1])
				}
				body, err := yaml.Marshal(document)
				if err != nil {
					t.Fatal(err)
				}
				if !raw {
					body, err = sigsyaml.YAMLToJSON(body)
					if err != nil {
						t.Fatal(err)
					}
				}
				previousPropagate := propagateConfig
				calls := 0
				propagateConfig = func(string, string) error { calls++; return nil }
				t.Cleanup(func() { propagateConfig = previousPropagate })
				request := httptest.NewRequest(http.MethodPut, "/", bytes.NewReader(body))
				request = request.WithContext(requestAs("/", "config.write").Context())
				result := httptest.NewRecorder()
				update(result, request)
				if result.Code != http.StatusBadRequest {
					t.Errorf("retargeted save status=%d, want 400: %s", result.Code, result.Body.String())
				}
				saved, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(saved, original) {
					t.Error("rejected target change modified persisted configuration")
				}
				if calls != 0 {
					t.Error("rejected target change propagated configuration to the runtime")
				}
			})
		}
	}
}

func globalSecretEditorHandlers(path string, raw bool) (http.HandlerFunc, http.HandlerFunc) {
	if raw {
		return GlobalConfigYAMLHandler(path), UpdateGlobalConfigYAMLHandler(path, false, filepath.Dir(path))
	}
	return RouterDefaultsHandler(path), UpdateRouterDefaultsHandler(path, false, filepath.Dir(path))
}
