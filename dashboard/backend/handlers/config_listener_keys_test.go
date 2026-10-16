package handlers

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

const listenerKeyFixture = "listener-client-key-fixture"

func writeKeyedListenerConfig(t *testing.T) string {
	t.Helper()
	path := createValidTestConfig(t, t.TempDir())
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	keyed := strings.Replace(string(data), "    port: 8801\n", "    port: 8801\n    api_keys:\n      - "+listenerKeyFixture+"\n", 1)
	if keyed == string(data) {
		t.Fatal("fixture listener not found")
	}
	keyedPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(keyedPath, []byte(keyed), 0o600); err != nil {
		t.Fatal(err)
	}
	return keyedPath
}

func requestAs(path string, perms ...string) *http.Request {
	request := httptest.NewRequest(http.MethodGet, path, nil)
	if perms == nil {
		return request
	}
	granted := map[string]bool{}
	for _, perm := range perms {
		granted[perm] = true
	}
	return request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{UserID: "user", Perms: granted}))
}

func TestConfigReadEndpointsHideListenerKeysFromCallersWhoCannotWriteConfig(t *testing.T) {
	configPath := writeKeyedListenerConfig(t)
	cases := []struct {
		name     string
		readonly bool
		perms    []string
		visible  bool
	}{
		{name: "read role", perms: []string{auth.PermConfigRead}},
		{name: "write role", perms: []string{auth.PermConfigRead, auth.PermConfigWrite}, visible: true},
		{name: "deploy-only grant", perms: []string{auth.PermConfigRead, auth.PermConfigDeploy}, visible: true},
		{name: "no auth context"},
		{name: "write role on readonly dashboard", readonly: true, perms: []string{auth.PermConfigRead, auth.PermConfigWrite}},
	}
	for _, tc := range cases {
		for _, endpoint := range []struct {
			path    string
			handler http.HandlerFunc
		}{
			{"/api/router/config/all", ConfigHandler(configPath, tc.readonly)},
			{"/api/router/config/yaml", ConfigYAMLHandler(configPath, tc.readonly)},
		} {
			t.Run(tc.name+" "+endpoint.path, func(t *testing.T) {
				response := httptest.NewRecorder()
				endpoint.handler(response, requestAs(endpoint.path, tc.perms...))
				if response.Code != http.StatusOK {
					t.Fatalf("status = %d: %s", response.Code, response.Body.String())
				}
				body := response.Body.String()
				if got := strings.Contains(body, listenerKeyFixture); got != tc.visible {
					t.Fatalf("listener key visible = %v, want %v:\n%s", got, tc.visible, body)
				}
				if !strings.Contains(body, "8801") || !strings.Contains(body, "default-business") {
					t.Fatalf("redacted response dropped unrelated config:\n%s", body)
				}
			})
		}
	}
}

func TestConfigYAMLReturnsOriginalBytesWhenNoListenerHasKeys(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	original, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	ConfigYAMLHandler(configPath, false)(response, requestAs("/api/router/config/yaml", auth.PermConfigRead))
	if response.Body.String() != string(original) {
		t.Fatalf("config bytes changed without listener keys:\n%s", response.Body.String())
	}
}

func TestConfigYAMLNeverServesListenerKeysHiddenBehindYAMLIndirection(t *testing.T) {
	documents := map[string]string{
		"alias listener": "version: v0.3\nx-gated: &gated\n  name: a\n  port: 8899\n  api_keys: [" + listenerKeyFixture + "]\nlisteners:\n  - *gated\n",
		"merge key":      "version: v0.3\nx-gated: &gated\n  port: 8899\n  api_keys: [" + listenerKeyFixture + "]\nlisteners:\n  - <<: *gated\n    name: b\n",
		"duplicate key":  "version: v0.3\nlisteners:\n  - name: a\n    port: 8899\n    api_keys: [first]\n    api_keys: [" + listenerKeyFixture + "]\n",
		"alias keys":     "version: v0.3\nx-keys: &keys [" + listenerKeyFixture + "]\nlisteners:\n  - name: a\n    port: 8899\n    api_keys: *keys\n",
		"alias key item": "version: v0.3\nx-key: &key " + listenerKeyFixture + "\nlisteners:\n  - name: a\n    port: 8899\n    api_keys: [*key]\n",
	}
	for name, document := range documents {
		t.Run(name, func(t *testing.T) {
			configPath := filepath.Join(t.TempDir(), "config.yaml")
			if err := os.WriteFile(configPath, []byte(document), 0o600); err != nil {
				t.Fatal(err)
			}
			response := httptest.NewRecorder()
			ConfigYAMLHandler(configPath, false)(response, requestAs("/api/router/config/yaml", auth.PermConfigRead))
			if strings.Contains(response.Body.String(), listenerKeyFixture) {
				t.Fatalf("listener key leaked (status %d):\n%s", response.Code, response.Body.String())
			}
		})
	}
}
