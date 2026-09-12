package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func sparseTransportGlobalCases() []struct {
	name   string
	global any
} {
	return []struct {
		name   string
		global any
	}{
		{name: "empty", global: map[string]any{}},
		{name: "router strategy", global: map[string]any{"router": map[string]any{"strategy": "priority"}}},
		{name: "service field", global: map[string]any{"services": map[string]any{"response_api": map[string]any{"ttl_seconds": 321}}}},
		{name: "explicit false", global: map[string]any{
			"router":   map[string]any{"clear_route_cache": false, "model_selection": map[string]any{"enabled": false}},
			"services": map[string]any{"response_api": map[string]any{"enabled": false}},
		}},
	}
}

func assertTransportGlobalParity(t *testing.T, original, roundTrip []byte) {
	t.Helper()
	want, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(original)
	if err != nil {
		t.Fatalf("parse original document: %v", err)
	}
	got, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(roundTrip)
	if err != nil {
		t.Fatalf("parse transported document: %v", err)
	}
	wantGlobal := routerconfig.CanonicalGlobalFromRouterConfig(want)
	gotGlobal := routerconfig.CanonicalGlobalFromRouterConfig(got)
	if !reflect.DeepEqual(wantGlobal, gotGlobal) {
		wantYAML, _ := yaml.Marshal(wantGlobal)
		gotYAML, _ := yaml.Marshal(gotGlobal)
		t.Fatalf("transport changed effective global config\nwant:\n%s\ngot:\n%s", wantYAML, gotYAML)
	}
}

func requireTransportHTTPResult(t *testing.T, handler http.HandlerFunc, path string, body []byte) *httptest.ResponseRecorder {
	t.Helper()
	response := httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("%s returned %d: %s", path, response.Code, response.Body.String())
	}
	return response
}

// Exercise the same import -> validate -> activate HTTP chain used by setup,
// checking the Router's effective config at each serialized boundary. Testing
// only the decoder or a zero-valued Go struct misses the original misrouting.
func TestSetupSparseGlobalRoundTrip(t *testing.T) {
	isolateConfigMutationRuntime(t)
	for _, tc := range sparseTransportGlobalCases() {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			configPath := createBootstrapSetupConfig(t, dir)
			resolver := setupmode.New(configPath, false)
			patch := createValidSetupPatch()
			patch["global"] = tc.global
			original := mustJSONRaw(t, patch)
			remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write(original)
			}))
			defer remote.Close()

			imported := requireTransportHTTPResult(t, SetupImportRemoteHandler(configPath, resolver), "/api/setup/import-remote",
				mustJSONRaw(t, SetupImportRemoteRequest{URL: remote.URL}))
			var importResult SetupImportRemoteResponse
			if err := json.Unmarshal(imported.Body.Bytes(), &importResult); err != nil {
				t.Fatal(err)
			}
			assertTransportGlobalParity(t, original, importResult.Config)

			validated := requireTransportHTTPResult(t, SetupValidateHandler(configPath, resolver), "/api/setup/validate",
				mustJSONRaw(t, SetupConfigRequest{Config: importResult.Config}))
			var validateResult SetupValidateResponse
			if err := json.Unmarshal(validated.Body.Bytes(), &validateResult); err != nil {
				t.Fatal(err)
			}
			if !validateResult.Valid || !validateResult.CanActivate {
				t.Fatalf("imported config must be activatable: %+v", validateResult)
			}
			assertTransportGlobalParity(t, original, validateResult.Config)

			requireTransportHTTPResult(t, SetupActivateHandler(configPath, false, dir, resolver), "/api/setup/activate",
				mustJSONRaw(t, SetupConfigRequest{Config: validateResult.Config}))
			activated, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatal(err)
			}
			assertTransportGlobalParity(t, original, activated)
			if resolver.Active() {
				t.Fatal("activation did not leave setup mode")
			}
		})
	}
}

func TestConfigEditorSparseGlobalRoundTrip(t *testing.T) {
	isolateConfigMutationRuntime(t)
	for _, tc := range sparseTransportGlobalCases() {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			configPath := createValidTestConfig(t, dir)
			config := canonicalConfigBody("127.0.0.1:8000")
			config["global"] = tc.global
			original := mustJSONRaw(t, config)
			if err := os.WriteFile(configPath, original, 0o600); err != nil {
				t.Fatal(err)
			}
			read := httptest.NewRecorder()
			ConfigHandler(configPath)(read, httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil))
			if read.Code != http.StatusOK {
				t.Fatalf("read config: %s", read.Body.String())
			}
			assertTransportGlobalParity(t, original, read.Body.Bytes())
			for _, payload := range [][]byte{original, read.Body.Bytes()} {
				requireTransportHTTPResult(t, UpdateConfigHandler(configPath, false, ""), "/api/router/config/update", payload)
				saved, err := os.ReadFile(configPath)
				if err != nil {
					t.Fatal(err)
				}
				assertTransportGlobalParity(t, original, saved)
			}
		})
	}
}

func TestSetupPatchWithoutGlobalPreservesExistingOverride(t *testing.T) {
	configPath := createBootstrapSetupConfig(t, t.TempDir())
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, []byte("global:\n  router:\n    clear_route_cache: false\n")...)
	if writeErr := os.WriteFile(configPath, data, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	candidate, err := buildSetupCandidateConfig(configPath,
		bytes.NewReader(mustJSONRaw(t, SetupConfigRequest{Config: mustJSONRaw(t, createValidSetupPatch())})),
		setupmode.New(configPath, false))
	if err != nil {
		t.Fatal(err)
	}
	if candidate.Global == nil || candidate.Global.Router.ClearRouteCache {
		t.Fatal("omitting global in a setup patch must retain the existing explicit false")
	}
	if !candidate.Global.Services.ResponseAPI.Enabled {
		t.Fatal("existing sparse global must still inherit unrelated service defaults")
	}
}

func TestConfigTransportPreservesGlobalEnvironmentReferences(t *testing.T) {
	t.Setenv("TRANSPORT_MODEL_PATH", "resolved-path-must-not-be-persisted")
	input := []byte("global:\n  model_catalog:\n    embeddings:\n      semantic:\n        qwen3_model_path: ${TRANSPORT_MODEL_PATH}\n")
	config, err := decodeYAMLTaggedBytes[canonicalConfigTransport](input)
	if err != nil {
		t.Fatal(err)
	}
	if config.Global.ModelCatalog.Embeddings.Semantic.Qwen3ModelPath != "${TRANSPORT_MODEL_PATH}" {
		t.Fatal("transport expanded an environment reference into persisted configuration")
	}
}

func TestConfigTransportResolvesGlobalAliasesFromOtherSections(t *testing.T) {
	input := []byte("version: v0.3\nlisteners: &empty []\nglobal:\n  model_catalog:\n    kbs: *empty\n")
	config, err := decodeYAMLTaggedBytes[canonicalConfigTransport](input)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := marshalYAMLBytes(config)
	if err != nil {
		t.Fatal(err)
	}
	assertTransportGlobalParity(t, input, roundTrip)
}
