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
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	if refreshErr := store.RefreshFromCanonical(configprojection.RefreshInput{
		Version: "secret-fixture", Source: configprojection.SourceDSL,
		YAMLBytes: data, DSLSnapshot: configSecretFixture,
	}); refreshErr != nil {
		t.Fatal(refreshErr)
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

func TestConfigSecretViewsPreserveDelegatedPolicy(t *testing.T) {
	for _, globalOnly := range []bool{false, true} {
		for _, raw := range []bool{false, true} {
			for _, privileged := range []bool{false, true} {
				name := map[bool]string{false: "full", true: "global"}[globalOnly] + "/" +
					map[bool]string{false: "json", true: "yaml"}[raw] + "/" +
					map[bool]string{false: "editor", true: "secret viewer"}[privileged]
				t.Run(name, func(t *testing.T) {
					path := writeSecretConfig(t, configSecretFixture)
					original, err := os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
					original = append(original, []byte("    authz:\n      delegated_credential:\n        enabled: true\n        header: x-reviewed-credential\n")...)
					if writeErr := os.WriteFile(path, original, 0o600); writeErr != nil {
						t.Fatal(writeErr)
					}
					read := ConfigHandler(path, false)
					if raw {
						read = ConfigYAMLHandler(path, false)
					}
					update := UpdateConfigHandler(path, false, filepath.Dir(path))
					if globalOnly {
						read, update = globalSecretEditorHandlers(path, raw)
					}
					permissions := []string{"config.read", "config.write"}
					if privileged {
						permissions = append(permissions, configSecretPermission)
					}
					view := httptest.NewRecorder()
					read(view, requestAs("/", permissions...))
					if view.Code != http.StatusOK {
						t.Fatalf("read status=%d: %s", view.Code, view.Body.String())
					}
					if visible := strings.Contains(view.Body.String(), configSecretFixture); visible != privileged {
						t.Fatalf("plaintext credential visible=%v", visible)
					}
					document, err := configSecretDocument(view.Body.Bytes())
					if err != nil {
						t.Fatal(err)
					}
					if !globalOnly {
						document = mappingValueNode(document, "global")
					}
					policy := mappingValueNode(mappingValueNode(mappingValueNode(document, "services"), "authz"), "delegated_credential")
					if enabled := mappingValueNode(policy, "enabled"); enabled == nil || enabled.Tag != "!!bool" || enabled.Value != "true" {
						t.Error("read view changed the delegated policy enabled type/value")
					}
					if header := mappingValueNode(policy, "header"); header == nil || header.Value != "x-reviewed-credential" {
						t.Error("read view changed the delegated policy header name")
					}
					previousPropagate := propagateConfig
					propagateConfig = func(string, string) error { return nil }
					t.Cleanup(func() { propagateConfig = previousPropagate })
					request := httptest.NewRequest(http.MethodPut, "/", bytes.NewReader(view.Body.Bytes()))
					request = request.WithContext(requestAs("/", permissions...).Context())
					result := httptest.NewRecorder()
					update(result, request)
					if result.Code != http.StatusOK {
						t.Fatalf("unchanged save status=%d: %s", result.Code, result.Body.String())
					}
					saved, err := os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
					if bytes.Contains(saved, []byte(redactedConfigSecret)) {
						t.Fatal("saved config contains redaction placeholders")
					}
					parsed, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(saved)
					if err != nil {
						t.Fatal(err)
					}
					if !parsed.Authz.DelegatedCredential.Enabled || parsed.Authz.DelegatedCredential.Header != "x-reviewed-credential" || parsed.ResponseAPI.Redis.Password != configSecretFixture {
						t.Fatal("round trip changed policy or stored credentials")
					}
				})
			}
		}
	}
}
