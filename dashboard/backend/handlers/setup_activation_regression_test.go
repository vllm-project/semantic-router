package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func setupActivationRegressionRuntime(t *testing.T) (string, string, func() []string) {
	t.Helper()
	root := t.TempDir()
	stateDir := filepath.Join(root, ".vllm-sr")
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		t.Fatal(err)
	}
	configPath := createBootstrapSetupConfig(t, stateDir)
	containerCLICalls := trapContainerCLIs(t)
	t.Setenv(routerContainerNameEnv, "activation-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "activation-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "activation-vllm-sr-dashboard-container")
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", configPath)
	t.Setenv(routerconfig.ManagementInternalListenerEnv, "true")
	t.Setenv("VLLM_SR_ENVOY_CONFIG_PATH", filepath.Join(root, "envoy.yaml"))
	t.Setenv("VLLM_SR_PYTHON_BIN", testRuntimeSyncPythonBinary(t))
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_CLI_PATH", filepath.Join(repoRoot, "src", "vllm-sr"))
	return configPath, root, containerCLICalls
}

func TestSetupActivationRendersOmittedReasoningAndRecordsTheActivation(t *testing.T) {
	configPath, root, containerCLICalls := setupActivationRegressionRuntime(t)
	patch := createValidSetupPatch()
	routing := patch["routing"].(map[string]interface{})
	decisions := routing["decisions"].([]map[string]interface{})
	refs := decisions[0]["modelRefs"].([]map[string]interface{})
	delete(refs[0], "use_reasoning")
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, patch)})
	w := httptest.NewRecorder()
	SetupActivateHandler(configPath, false, root, setupmode.New(configPath, false))(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("activation failed: %d %s", w.Code, w.Body.String())
	}
	// The CLI that owns the stack starts the services; setup touches no container.
	if calls := containerCLICalls(); len(calls) != 0 {
		t.Fatalf("setup ran a container CLI: %q", calls)
	}
	if _, err := os.Stat(pendingActivationPath(configPath, pendingActivationSuffix)); err != nil {
		t.Fatalf("activation was not recorded: %v", err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil || bytes.Contains(data, []byte("use_reasoning: null")) {
		t.Fatalf("activated config is not canonical: %s, %v", data, err)
	}
	active, err := routerconfig.Parse(configPath)
	if err != nil || active.ManagementAPI.BindAddress != "0.0.0.0" || active.ManagementAPI.Port != 8080 {
		t.Fatalf("setup did not materialize the container management listener: %+v, %v", active, err)
	}
	if _, statErr := os.Stat(filepath.Join(filepath.Dir(configPath), ".vllm-sr", "runtime-config.yaml")); !os.IsNotExist(statErr) {
		t.Fatalf("activation nested the runtime-owned config: %v", statErr)
	}
	if _, statErr := os.Stat(filepath.Join(root, "envoy.yaml")); !os.IsNotExist(statErr) {
		t.Fatalf("setup rendered the Envoy configuration the CLI owns: %v", statErr)
	}
}

func TestSetupActivationMaterializesAuthoredEmptyListener(t *testing.T) {
	configPath, root, _ := setupActivationRegressionRuntime(t)
	patch := createValidSetupPatch()
	// An authored empty listener retains its omission semantics through the
	// shared managed CLI materializer, without typed-transport cleanup.
	patch["global"] = map[string]interface{}{
		"services": map[string]interface{}{"management_api": map[string]interface{}{}},
	}
	w := httptest.NewRecorder()
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, patch)})
	SetupActivateHandler(configPath, false, root, setupmode.New(configPath, false))(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("activation failed: %d %s", w.Code, w.Body.String())
	}
	active, err := routerconfig.Parse(configPath)
	if err != nil || active.ManagementAPI.BindAddress != "0.0.0.0" || active.ManagementAPI.Port != 8080 {
		t.Fatalf("authored empty listener suppressed runtime materialization: %+v, %v", active, err)
	}
}

func TestSetupActivationPreservesExplicitManagementListener(t *testing.T) {
	configPath, root, _ := setupActivationRegressionRuntime(t)
	patch := createValidSetupPatch()
	patch["global"] = map[string]interface{}{
		"services": map[string]interface{}{
			"management_api": map[string]interface{}{
				"bind_address": "0.0.0.0",
				"port":         9091,
				"auth": map[string]interface{}{
					"mode":   "bearer",
					"tokens": []map[string]interface{}{{"env": "TEST_MANAGEMENT_TOKEN", "role": "admin"}},
				},
			},
		},
	}
	w := httptest.NewRecorder()
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, patch)})
	SetupActivateHandler(configPath, false, root, setupmode.New(configPath, false))(
		w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("activation failed: %d %s", w.Code, w.Body.String())
	}
	active, err := routerconfig.Parse(configPath)
	if err != nil {
		t.Fatal(err)
	}
	listener := active.ManagementAPI
	if listener.BindAddress != "0.0.0.0" || listener.Port != 9091 || listener.Auth.Mode != "bearer" ||
		len(listener.Auth.Tokens) != 1 || listener.Auth.Tokens[0].Env != "TEST_MANAGEMENT_TOKEN" {
		t.Fatalf("explicit listener/auth was changed: %+v", listener)
	}
}

func TestSetupActivationFailureRestoresBootstrapAndAllowsRetry(t *testing.T) {
	configPath, root, _ := setupActivationRegressionRuntime(t)
	previous, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	// A directory where the activation record goes: the record cannot be written.
	record := pendingActivationPath(configPath, pendingActivationSuffix)
	if mkdirErr := os.MkdirAll(filepath.Join(record, "occupied"), 0o700); mkdirErr != nil {
		t.Fatal(mkdirErr)
	}
	resolver := setupmode.New(configPath, false)
	handler := SetupActivateHandler(configPath, false, root, resolver)
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	w := httptest.NewRecorder()
	handler(w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("an unrecorded activation must not report success: %d %s", w.Code, w.Body.String())
	}
	var response map[string]interface{}
	if decodeErr := json.Unmarshal(w.Body.Bytes(), &response); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if response["configuration_restored"] != true || response["setupMode"] != true || response["stage"] != "activation_record" {
		t.Fatalf("failure was not retryable: %#v", response)
	}
	restored, err := os.ReadFile(configPath)
	if err != nil || !bytes.Equal(restored, previous) || !resolver.Resolve().Active {
		t.Fatalf("bootstrap was not restored: %s, %v", restored, err)
	}
	if removeErr := os.RemoveAll(record); removeErr != nil {
		t.Fatal(removeErr)
	}
	w = httptest.NewRecorder()
	handler(w, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if w.Code != http.StatusOK || resolver.Resolve().Active {
		t.Fatalf("same activation could not recover: %d %s", w.Code, w.Body.String())
	}
	if _, statErr := os.Stat(record); statErr != nil {
		t.Fatalf("retry did not record the activation: %v", statErr)
	}
}

func TestSetupActivationReportsRollbackFailureWithoutRawError(t *testing.T) {
	root := t.TempDir()
	configPath := createBootstrapSetupConfig(t, root)
	resolver := setupmode.New(configPath, false)
	originalRename := atomicRename
	t.Cleanup(func() { atomicRename = originalRename })
	atomicRename = func(string, string) error { return errors.New("secret-provider-credential") }
	w := httptest.NewRecorder()
	failSetupActivation(w, configPath, []byte("setup:\n  mode: true\n"), resolver, "activation_record")
	if w.Code != http.StatusInternalServerError || strings.Contains(w.Body.String(), "secret-provider-credential") {
		t.Fatalf("unsafe failure response: %d %s", w.Code, w.Body.String())
	}
	var response map[string]interface{}
	if decodeErr := json.Unmarshal(w.Body.Bytes(), &response); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if response["configuration_restored"] != false {
		t.Fatalf("failed rollback claimed restoration: %#v", response)
	}
}
