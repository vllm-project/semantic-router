package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const providerKeyFixture = "sk-test-keyref-123"

func saveConfigWithBackendKey(t *testing.T, configPath, key string) {
	t.Helper()
	body := canonicalConfigBody("127.0.0.1:8000")
	body["global"] = map[string]interface{}{}
	models := body["providers"].(map[string]interface{})["models"].([]map[string]interface{})
	models[0]["backend_refs"].([]map[string]interface{})[0]["api_key"] = key
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	UpdateConfigHandler(configPath, false, "")(response, httptest.NewRequest(http.MethodPost, "/api/router/config/update", bytes.NewReader(data)))
	if response.Code != http.StatusOK {
		t.Fatalf("save status = %d: %s", response.Code, response.Body.String())
	}
}

// savedKeyFile returns the file the saved config names for the backend key,
// after checking the config holds no literal key.
func savedKeyFile(t *testing.T, configPath, key string) string {
	t.Helper()
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte(key)) || bytes.Contains(data, []byte("api_key:")) {
		t.Fatalf("the saved config holds the key:\n%s", data)
	}
	cfg, err := routerconfig.ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.VLLMEndpoints[0].APIKey != key {
		t.Fatal("the Router does not resolve the stored key")
	}
	auth := cfg.ModelConfig["test-model"].AuthoredModel.BackendRefs[0].Auth
	if auth == nil || filepath.Dir(auth.APIKeyFile) != filepath.Join(filepath.Dir(configPath), "provider-keys") {
		t.Fatalf("auth = %+v, want a file in the provider key directory", auth)
	}
	return auth.APIKeyFile
}

func TestSavingABackendKeyStoresItInAnOwnerOnlyFile(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	saveConfigWithBackendKey(t, configPath, providerKeyFixture)

	path := savedKeyFile(t, configPath, providerKeyFixture)
	for name, want := range map[string]os.FileMode{path: 0o600, filepath.Dir(path): 0o700} {
		info, err := os.Stat(name)
		if err != nil || info.Mode().Perm() != want {
			t.Fatalf("%s mode = %v (%v), want %v", name, info.Mode().Perm(), err, want)
		}
	}
}

func TestSavingKeepsAnEnvironmentReference(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	saveConfigWithBackendKey(t, configPath, "${OPENAI_API_KEY}")

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "api_key: ${OPENAI_API_KEY}") || strings.Contains(string(data), "api_key_file") {
		t.Fatalf("an environment reference must stay as written:\n%s", data)
	}
}

func TestTheNextSaveMovesALiteralKeyOut(t *testing.T) {
	configPath := createValidTestConfig(t, t.TempDir())
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	literal := strings.Replace(string(data), "weight: 1\n", "weight: 1\n          api_key: "+providerKeyFixture+"\n", 1) + "global: {}\n"
	if err := os.WriteFile(configPath, []byte(literal), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := routerconfig.ParseYAMLBytes([]byte(literal)); err != nil {
		t.Fatalf("a config with a literal key no longer loads: %v", err)
	}

	// The editor sends the masked placeholder back for the key it never saw.
	saveConfigWithBackendKey(t, configPath, redactedConfigSecret)
	savedKeyFile(t, configPath, providerKeyFixture)
}
