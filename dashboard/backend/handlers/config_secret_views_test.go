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

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

func TestConfigSecretViewsCoverHistoricalProjectionsAndPreview(t *testing.T) {
	path := writeSecretConfig(t, configSecretFixture)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	store := openTestConfigProjectionStore(t)
	SetConfigProjectionStore(store)
	t.Cleanup(func() { SetConfigProjectionStore(nil) })
	if err := store.RefreshFromCanonical(configprojection.RefreshInput{
		Version: "secret-fixture", Source: configprojection.SourceDSL,
		YAMLBytes: data, DSLSnapshot: configSecretFixture,
	}); err != nil {
		t.Fatal(err)
	}
	for _, endpoint := range []struct {
		path    string
		handler http.HandlerFunc
	}{
		{"/api/router/config/deployments/secret-fixture", ConfigDeploymentDetailHandler()},
		{"/api/router/config/active-projection", ActiveConfigProjectionHandler()},
	} {
		for _, privileged := range []bool{false, true} {
			permissions := []string{"config.read"}
			if privileged {
				permissions = append(permissions, configSecretPermission)
			}
			response := httptest.NewRecorder()
			endpoint.handler(response, requestAs(endpoint.path, permissions...))
			if response.Code != 200 || strings.Contains(response.Body.String(), configSecretFixture) != privileged {
				t.Errorf("projection %s privileged=%v status=%d", endpoint.path, privileged, response.Code)
			}
		}
	}
	body, err := json.Marshal(DeployRequest{YAML: "routing: {}\n"})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/router/config/deploy/preview", bytes.NewReader(body))
	request = request.WithContext(requestAs("/", "config.read", "config.deploy").Context())
	response := httptest.NewRecorder()
	DeployPreviewHandler(path, false)(response, request)
	if response.Code != 200 || strings.Contains(response.Body.String(), configSecretFixture) {
		t.Fatalf("preview exposed a secret or failed: status=%d", response.Code)
	}
}

func TestConfigSecretFullEditorRoundTrip(t *testing.T) {
	for _, raw := range []bool{false, true} {
		path := writeSecretConfig(t, configSecretFixture)
		read := ConfigHandler(path, false)
		if raw {
			read = ConfigYAMLHandler(path, false)
		}
		view := httptest.NewRecorder()
		read(view, requestAs("/", "config.read", "config.write"))
		if view.Code != 200 || strings.Contains(view.Body.String(), configSecretFixture) {
			t.Fatal("full editor received a plaintext credential")
		}
		previous := propagateConfig
		propagateConfig = func(string, string) error { return nil }
		request := httptest.NewRequest(http.MethodPut, "/", bytes.NewReader(view.Body.Bytes()))
		result := httptest.NewRecorder()
		UpdateConfigHandler(path, false, filepath.Dir(path))(result, request)
		propagateConfig = previous
		if result.Code != 200 {
			t.Fatalf("full editor save status=%d: %s", result.Code, result.Body.String())
		}
		persisted, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(persisted, []byte(configSecretFixture)) || bytes.Contains(persisted, []byte(redactedConfigSecret)) {
			t.Fatal("full editor damaged the stored credential")
		}
		merged, err := mergeDeployPayload(persisted, DeployRequest{YAML: "routing: {}\n", BaseYAML: view.Body.String()})
		if err != nil || !bytes.Contains(merged, []byte(configSecretFixture)) {
			t.Fatalf("deploy from redacted imported base lost credential: %v", err)
		}
	}
}
