package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

// postConfig saves body through the config editor. A request with no session
// may not read configuration secrets, as a Write-role user may not; perms are
// the session's permissions otherwise.
func postConfig(t *testing.T, configPath string, body map[string]interface{}, perms ...string) *httptest.ResponseRecorder {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(data))
	if len(perms) > 0 {
		granted := map[string]bool{}
		for _, perm := range perms {
			granted[perm] = true
		}
		request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{Perms: granted}))
	}
	response := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, filepath.Dir(configPath))(response, request)
	return response
}

// backendBody is the saved config with fields set on its backend at
// endpoint, then the extra backends.
func backendBody(endpoint string, fields map[string]interface{}, extra ...map[string]interface{}) map[string]interface{} {
	body := canonicalConfigBody(endpoint)
	body["global"] = map[string]interface{}{}
	model := body["providers"].(map[string]interface{})["models"].([]map[string]interface{})[0]
	refs := model["backend_refs"].([]map[string]interface{})
	for field, value := range fields {
		refs[0][field] = value
	}
	model["backend_refs"] = append(refs, extra...)
	return body
}

func replicaBackend(endpoint string, fields map[string]interface{}) map[string]interface{} {
	backend := map[string]interface{}{"name": "endpoint2", "provider": "vllm", "endpoint": endpoint, "protocol": "http", "weight": 1}
	for field, value := range fields {
		backend[field] = value
	}
	return backend
}

func keyFile(path string) map[string]interface{} {
	return map[string]interface{}{"auth": map[string]interface{}{"api_key_file": path}}
}

func requireUnchangedFile(t *testing.T, path string, before []byte) {
	t.Helper()
	after, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("%s changed (%v):\n%s", path, err, after)
	}
}

// A Write-role user sees a key file's path, never the key, so moving the
// backend that names it would send the key to a host of their choosing.
func TestAWriterCannotSendASavedKeyToAnotherHost(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	saveConfigWithBackendKey(t, configPath, providerKeyFixture)
	path := savedKeyFile(t, configPath, providerKeyFixture)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	moved := backendBody("attacker.example:8000", keyFile(path))
	for _, test := range []struct {
		name string
		body map[string]interface{}
	}{
		{"moved", moved},
		{"named by a new backend", backendBody("127.0.0.1:8000", keyFile(path), replicaBackend("attacker.example:8000", keyFile(path)))},
	} {
		if response := postConfig(t, configPath, test.body); response.Code != http.StatusForbidden {
			t.Fatalf("%s: status = %d, want 403: %s", test.name, response.Code, response.Body.String())
		}
		requireUnchangedFile(t, configPath, before)
	}

	if response := postConfig(t, configPath, backendBody("127.0.0.1:8000", keyFile(path))); response.Code != http.StatusOK {
		t.Fatalf("a save that keeps the backend where it was: status = %d: %s", response.Code, response.Body.String())
	}
	if response := postConfig(t, configPath, moved, auth.PermConfigSecretsRead); response.Code != http.StatusOK {
		t.Fatalf("a user who may read secrets: status = %d: %s", response.Code, response.Body.String())
	}
}

// Nor can a Write-role user name a key file the config does not use: the
// Router would send whatever it reads there, another key included.
func TestAWriterCannotNameAKeyFileTheConfigDoesNotUse(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	saveConfigWithBackendKey(t, configPath, providerKeyFixture)
	path := savedKeyFile(t, configPath, providerKeyFixture)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	unnamed := filepath.Join(filepath.Dir(path), "key-earlier")
	outside := filepath.Join(t.TempDir(), "other.key")
	for _, file := range []string{unnamed, outside} {
		if err = os.WriteFile(file, []byte("sk-other-secret"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	for _, file := range []string{unnamed, outside, "/proc/self/environ"} {
		if response := postConfig(t, configPath, backendBody("127.0.0.1:8000", keyFile(file))); response.Code != http.StatusForbidden {
			t.Fatalf("%s: status = %d, want 403: %s", file, response.Code, response.Body.String())
		}
		requireUnchangedFile(t, configPath, before)
	}
	if response := postConfig(t, configPath, backendBody("127.0.0.1:8000", keyFile(outside)), auth.PermConfigSecretsRead); response.Code != http.StatusOK {
		t.Fatalf("a user who may read secrets: status = %d: %s", response.Code, response.Body.String())
	}
}

// On Kubernetes a save moves a typed key into the chart's key Secret and names
// its variable, which the Write role sees as plainly as a key file's path.
// Moving a backend that names one is refused the same way, while a variable
// the operator wrote stays theirs to point anywhere.
func TestAWriterCannotSendAnEnvironmentKeyToAnotherHost(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	for name, fields := range map[string]map[string]interface{}{
		"auth.api_key_env": {"auth": map[string]interface{}{"api_key_env": "VSR_KEY_TEST"}},
		"${NAME} api_key":  {"api_key": "${VSR_KEY_TEST}"},
	} {
		if response := postConfig(t, configPath, backendBody("127.0.0.1:8000", fields), auth.PermConfigSecretsRead); response.Code != http.StatusOK {
			t.Fatalf("%s: an admin save: status = %d: %s", name, response.Code, response.Body.String())
		}
		before, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatal(err)
		}
		if response := postConfig(t, configPath, backendBody("attacker.example:8000", fields)); response.Code != http.StatusForbidden {
			t.Fatalf("%s moved: status = %d, want 403: %s", name, response.Code, response.Body.String())
		}
		requireUnchangedFile(t, configPath, before)
		if response := postConfig(t, configPath, backendBody("127.0.0.1:8000", fields)); response.Code != http.StatusOK {
			t.Fatalf("%s kept in place: status = %d: %s", name, response.Code, response.Body.String())
		}
	}
	operator := map[string]interface{}{"auth": map[string]interface{}{"api_key_env": "OPENAI_API_KEY"}}
	if response := postConfig(t, configPath, backendBody("provider.example:8000", operator)); response.Code != http.StatusOK {
		t.Fatalf("an operator's variable: status = %d: %s", response.Code, response.Body.String())
	}
}
