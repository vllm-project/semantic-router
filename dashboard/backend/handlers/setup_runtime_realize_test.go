package handlers

import (
	"bytes"
	"encoding/json"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func configureSetupRuntimeCLI(t *testing.T) {
	t.Helper()
	t.Setenv(routerconfig.ManagementInternalListenerEnv, "true")
	t.Setenv("VLLM_SR_PYTHON_BIN", testRuntimeSyncPythonBinary(t))
	repo, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_CLI_PATH", filepath.Join(repo, "src", "vllm-sr"))
}

func managedSetupRuntime(t *testing.T) (string, *setupmode.Resolver) {
	t.Helper()
	root := t.TempDir()
	state := filepath.Join(root, ".vllm-sr")
	if err := os.MkdirAll(state, 0o700); err != nil {
		t.Fatal(err)
	}
	configPath := createBootstrapSetupConfig(t, state)
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", configPath)
	t.Setenv("VLLM_SR_STATE_ROOT_DIR", root)
	t.Setenv(routerContainerNameEnv, "setup-test-router")
	t.Setenv(dashboardContainerNameEnv, "setup-test-dashboard")
	t.Setenv(envoyContainerNameEnv, "setup-test-envoy")
	configureSetupRuntimeCLI(t)
	// Service orchestration is independently tested with lifecycle fixtures.
	// This test executes the real CLI materializer without contacting Docker.
	docker := filepath.Join(root, "docker")
	if err := os.WriteFile(docker, []byte("#!/bin/sh\nexit 1\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	return configPath, setupmode.New(configPath, false)
}

func importSetupRuntimePatch(t *testing.T, path string, resolver *setupmode.Resolver, patch map[string]interface{}) json.RawMessage {
	t.Helper()
	patch["version"] = "v0.3"
	raw, err := yaml.Marshal(patch)
	if err != nil {
		t.Fatal(err)
	}
	profile := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(raw)
	}))
	t.Cleanup(profile.Close)
	response := httptest.NewRecorder()
	SetupImportRemoteHandler(path, resolver)(response, httptest.NewRequest(http.MethodPost, "/api/setup/import-remote", bytes.NewReader(mustJSONRaw(t, SetupImportRemoteRequest{URL: profile.URL}))))
	if response.Code != http.StatusOK {
		t.Fatalf("import = %d: %s", response.Code, response.Body.String())
	}
	var imported SetupImportRemoteResponse
	if err := json.Unmarshal(response.Body.Bytes(), &imported); err != nil {
		t.Fatal(err)
	}
	return imported.Config
}

func TestManagedSetupRealizesImportedConfigBeforePublishingOwnedPath(t *testing.T) {
	configPath, resolver := managedSetupRuntime(t)
	patch := createValidSetupPatch()
	patch["global"] = map[string]interface{}{"services": map[string]interface{}{"router_replay": map[string]interface{}{"enabled": false}}}
	imported := importSetupRuntimePatch(t, configPath, resolver, patch)
	// The typed import roundtrip emits an empty management object. It must not
	// turn absent listener intent into an explicit standalone loopback bind.
	var document map[string]interface{}
	if err := json.Unmarshal(imported, &document); err != nil {
		t.Fatal(err)
	}
	management := document["global"].(map[string]interface{})["services"].(map[string]interface{})["management_api"]
	if len(management.(map[string]interface{})) != 0 {
		t.Fatalf("fixture no longer exercises empty management roundtrip: %#v", management)
	}
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	body := mustJSONRaw(t, SetupConfigRequest{Config: imported})
	validated := httptest.NewRecorder()
	SetupValidateHandler(configPath, resolver)(validated, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))
	if validated.Code != http.StatusOK {
		t.Fatalf("validate = %d: %s", validated.Code, validated.Body.String())
	}
	stillBootstrap, err := os.ReadFile(configPath)
	if err != nil || !bytes.Equal(before, stillBootstrap) || !resolver.Active() {
		t.Fatalf("validation modified bootstrap: %v", err)
	}
	activated := httptest.NewRecorder()
	SetupActivateHandler(configPath, false, filepath.Dir(filepath.Dir(configPath)), resolver)(activated, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if activated.Code != http.StatusOK {
		t.Fatalf("activate = %d: %s", activated.Code, activated.Body.String())
	}
	parsed, err := routerconfig.Parse(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ManagementAPI.BindAddress != "0.0.0.0" || parsed.ManagementAPI.Port != 8080 {
		t.Fatalf("published management listener = %s, want container-reachable 0.0.0.0:8080", parsed.ManagementAPI.ListenAddress())
	}
	if resolver.Active() {
		t.Fatal("successful activation retained setup mode")
	}
	if _, statErr := os.Stat(filepath.Join(filepath.Dir(configPath), ".vllm-sr")); !os.IsNotExist(statErr) {
		t.Fatalf("nested runtime state created: %v", statErr)
	}
}

func TestManagedSetupRealizationFailurePreservesBootstrap(t *testing.T) {
	configPath, resolver := managedSetupRuntime(t)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_PYTHON_BIN", filepath.Join(t.TempDir(), "python-missing"))
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	response := httptest.NewRecorder()
	SetupActivateHandler(configPath, false, filepath.Dir(filepath.Dir(configPath)), resolver)(response, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if response.Code == http.StatusOK {
		t.Fatalf("activation published without a working materializer: %s", response.Body.String())
	}
	after, err := os.ReadFile(configPath)
	if err != nil || !bytes.Equal(before, after) || !resolver.Active() {
		t.Fatalf("failed realization changed bootstrap: %v", err)
	}
}

func TestManagedSetupPreservesExplicitManagementAndMissingGlobalDefaults(t *testing.T) {
	explicitManagement := map[string]interface{}{
		"bind_address": "0.0.0.0", "port": 9099, "remote_exposure": true,
		"auth": map[string]interface{}{
			"mode": "bearer", "tokens": []map[string]interface{}{{"env": "SETUP_TEST_TOKEN", "role": "admin"}},
			"roles": map[string][]string{"admin": {"*"}},
		},
	}
	for _, explicit := range []bool{false, true} {
		name := "missing-global"
		if explicit {
			name = "explicit-management"
		}
		t.Run(name, func(t *testing.T) {
			configPath, resolver := managedSetupRuntime(t)
			t.Setenv("SETUP_TEST_TOKEN", "test-only-token")
			patch := createValidSetupPatch()
			if explicit {
				patch["global"] = map[string]interface{}{"services": map[string]interface{}{"management_api": explicitManagement}}
			}
			body := mustJSONRaw(t, SetupConfigRequest{Config: importSetupRuntimePatch(t, configPath, resolver, patch)})
			response := httptest.NewRecorder()
			SetupActivateHandler(configPath, false, filepath.Dir(filepath.Dir(configPath)), resolver)(response, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
			if response.Code != http.StatusOK {
				t.Fatalf("activate = %d: %s", response.Code, response.Body.String())
			}
			published, err := readSetupConfigFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			if explicit {
				var expected routerconfig.ManagementAPIConfig
				raw, marshalErr := yaml.Marshal(explicitManagement)
				if marshalErr != nil {
					t.Fatal(marshalErr)
				}
				if decodeErr := yaml.Unmarshal(raw, &expected); decodeErr != nil {
					t.Fatal(decodeErr)
				}
				if !reflect.DeepEqual(published.Global.Services.ManagementAPI, expected) {
					t.Fatalf("explicit management changed: got %+v want %+v", published.Global.Services.ManagementAPI, expected)
				}
			} else {
				defaults := routerconfig.DefaultCanonicalGlobal()
				if !reflect.DeepEqual(published.Global.ModelCatalog, defaults.ModelCatalog) {
					t.Fatal("CLI listener realization lost Router model defaults for absent global")
				}
				if published.Global.Services.ManagementAPI.BindAddress != "0.0.0.0" {
					t.Fatalf("Router defaults replaced realized listener: %+v", published.Global.Services.ManagementAPI)
				}
			}
		})
	}
}

func TestManagedSetupRejectsUnreachableExplicitListenerWithoutPublishing(t *testing.T) {
	configPath, resolver := managedSetupRuntime(t)
	patch := createValidSetupPatch()
	patch["global"] = map[string]interface{}{"services": map[string]interface{}{"management_api": map[string]interface{}{"bind_address": "127.0.0.1", "port": 8080}}}
	body := mustJSONRaw(t, SetupConfigRequest{Config: importSetupRuntimePatch(t, configPath, resolver, patch)})
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, handler := range []http.HandlerFunc{SetupValidateHandler(configPath, resolver), SetupActivateHandler(configPath, false, filepath.Dir(filepath.Dir(configPath)), resolver)} {
		response := httptest.NewRecorder()
		handler(response, httptest.NewRequest(http.MethodPost, "/api/setup", bytes.NewReader(body)))
		if response.Code != http.StatusBadRequest {
			t.Fatalf("unreachable explicit listener = %d: %s", response.Code, response.Body.String())
		}
		after, readErr := os.ReadFile(configPath)
		if readErr != nil || !bytes.Equal(before, after) || !resolver.Active() {
			t.Fatalf("rejected listener changed bootstrap: %v", readErr)
		}
	}
}

func TestStandaloneSetupKeepsRouterListenerDefaultsWithoutRealizer(t *testing.T) {
	root := t.TempDir()
	configPath := createBootstrapSetupConfig(t, root)
	t.Setenv(routerContainerNameEnv, "")
	t.Setenv(dashboardContainerNameEnv, "")
	t.Setenv("TARGET_ROUTER_API_URL", "")
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", configPath)
	t.Setenv("VLLM_SR_PYTHON_BIN", filepath.Join(root, "python-missing"))
	resolver := setupmode.New(configPath, false)
	body := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})
	response := httptest.NewRecorder()
	SetupActivateHandler(configPath, false, root, resolver)(response, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("standalone activate = %d: %s", response.Code, response.Body.String())
	}
	published, err := readSetupConfigFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(published.Global.Services.ManagementAPI, routerconfig.DefaultManagementAPIConfig()) {
		t.Fatalf("standalone Router listener defaults changed: %+v", published.Global.Services.ManagementAPI)
	}
}

func setupFilesystemSnapshot(t *testing.T, root string) map[string]string {
	t.Helper()
	directory, err := os.OpenRoot(root)
	if err != nil {
		t.Fatal(err)
	}
	defer directory.Close()

	files := make(map[string]string)
	err = fs.WalkDir(directory.FS(), ".", func(relative string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			files[relative] = "directory"
			return nil
		}
		raw, readErr := directory.ReadFile(relative)
		files[relative] = string(raw)
		return readErr
	})
	if err != nil {
		t.Fatal(err)
	}
	return files
}

func TestManagedSetupValidationKeepsReferencedKBAssetsUnchanged(t *testing.T) {
	configPath, resolver := managedSetupRuntime(t)
	workspace := filepath.Dir(filepath.Dir(configPath))
	source := filepath.Join(filepath.Dir(configPath), "remote-kb")
	if err := os.Mkdir(source, 0o700); err != nil {
		t.Fatal(err)
	}
	manifest := []byte(`{"labels":{"safe":{"description":"Safe content","exemplars":["hello world"]}}}`)
	if err := os.WriteFile(filepath.Join(source, "labels.json"), manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	patch := createValidSetupPatch()
	patch["global"] = map[string]interface{}{"model_catalog": map[string]interface{}{"kbs": []map[string]interface{}{{
		"name": "remote_kb", "source": map[string]interface{}{"path": "remote-kb/", "manifest": "labels.json"}, "threshold": 0.55,
	}}}}
	patch["routing"].(map[string]interface{})["signals"].(map[string]interface{})["kb"] = []map[string]interface{}{{
		"name": "safe_content", "kb": "remote_kb", "target": map[string]interface{}{"kind": "label", "value": "safe"}, "match": "best",
	}}
	body := mustJSONRaw(t, SetupConfigRequest{Config: importSetupRuntimePatch(t, configPath, resolver, patch)})
	before := setupFilesystemSnapshot(t, workspace)
	patch["routing"].(map[string]interface{})["signals"].(map[string]interface{})["kb"].([]map[string]interface{})[0]["target"].(map[string]interface{})["value"] = "missing-label"
	invalidBody := mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, patch)})
	rejected := httptest.NewRecorder()
	SetupActivateHandler(configPath, false, workspace, resolver)(rejected, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(invalidBody)))
	if rejected.Code != http.StatusBadRequest {
		t.Fatalf("invalid KB label activation = %d: %s", rejected.Code, rejected.Body.String())
	}
	if !reflect.DeepEqual(before, setupFilesystemSnapshot(t, workspace)) || !resolver.Active() {
		t.Fatal("invalid candidate created or changed runtime/KB files")
	}
	validated := httptest.NewRecorder()
	SetupValidateHandler(configPath, resolver)(validated, httptest.NewRequest(http.MethodPost, "/api/setup/validate", bytes.NewReader(body)))
	if validated.Code != http.StatusOK {
		t.Fatalf("validate referenced KB = %d: %s", validated.Code, validated.Body.String())
	}
	if !reflect.DeepEqual(before, setupFilesystemSnapshot(t, workspace)) || !resolver.Active() {
		t.Fatal("validation created or changed runtime/KB files")
	}
	var response SetupValidateResponse
	if err := json.Unmarshal(validated.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	// Activate the actual returned document so validation cannot discard the
	// original asset location needed by the subsequent bootstrap copy.
	activationBody := mustJSONRaw(t, SetupConfigRequest{Config: response.Config})
	activated := httptest.NewRecorder()
	SetupActivateHandler(configPath, false, workspace, resolver)(activated, httptest.NewRequest(http.MethodPost, "/api/setup/activate", bytes.NewReader(activationBody)))
	if activated.Code != http.StatusOK {
		t.Fatalf("activate validated KB = %d: %s", activated.Code, activated.Body.String())
	}
	published, err := readSetupConfigFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	kb := published.Global.ModelCatalog.KBs[0]
	if kb.Source.Path != "knowledge_bases/remote-kb/" {
		t.Fatalf("runtime KB reference = %q", kb.Source.Path)
	}
	copied, err := os.ReadFile(kb.Source.ResolveManifestPath(filepath.Dir(configPath)))
	if err != nil || !bytes.Equal(copied, manifest) {
		t.Fatalf("activated KB manifest does not match source: %v", err)
	}
}
