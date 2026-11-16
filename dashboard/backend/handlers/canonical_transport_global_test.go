package handlers

import (
	"bytes"
	"encoding/json"
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
	// The fixture server is on loopback; declare it the way an operator would
	// declare a real internal target (#1388).
	allowLoopbackForTest(t)
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
			ConfigHandler(configPath, false)(read, httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil))
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
	input := []byte("global:\n  stores:\n    memory:\n      milvus:\n        address: ${TRANSPORT_MODEL_PATH}\n")
	config, err := decodeYAMLTaggedBytes[canonicalConfigTransport](input)
	if err != nil {
		t.Fatal(err)
	}
	if config.Global.Stores.Memory.Milvus.Address != "${TRANSPORT_MODEL_PATH}" {
		t.Fatal("transport expanded an environment reference into persisted configuration")
	}
}

func TestConfigTransportResolvesGlobalAliasesFromOtherSections(t *testing.T) {
	input := []byte("version: v0.5\nlisteners: &empty []\nglobal:\n  router:\n    headers:\n      session_id_headers: *empty\n")
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

// A delegated deployment is valid only when a recipe asks it, which the
// global block alone cannot show (installation/security-hardening.md).
const delegatedDecisionConfig = `version: v0.5
listeners:
  - name: public
    address: 0.0.0.0
    port: 8801
providers:
  defaults:
    model: test-model
  models:
    - name: test-model
      provider_model_id: test-model
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http
recipes:
  - name: default
    signals:
      systemone:
        - name: support_team
          description: Which team should handle the request.
          deployment: team-decisions
          question:
            type: choice
            instructions: Which team should handle this request?
            choices:
              - key: billing
                description: Payments and refunds
              - key: engineering
                description: Outages and bugs
    decisions:
      - name: billing-route
        priority: 1
        rules:
          operator: AND
          conditions:
            - type: systemone
              name: support_team
              label: billing
        modelRefs:
          - model: test-model
global:
  services:
    authz:
      delegated_credential:
        enabled: true
  model_catalog:
    deployments:
      team-decisions:
        provider: systemone
        endpoint: https://decisions.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth:
          credential: delegated
`

func requireDelegatedDeployment(t *testing.T, surface string, document []byte) {
	t.Helper()
	cfg, err := routerconfig.ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("%s: Router refuses the config: %v", surface, err)
	}
	if auth := cfg.ModelDeployments["team-decisions"].Auth; auth == nil || auth.Credential != routerconfig.CredentialDelegated {
		t.Fatalf("%s: team-decisions auth = %+v, want credential %q", surface, auth, routerconfig.CredentialDelegated)
	}
}

func TestConfigEditorKeepsADelegatedDecisionDeployment(t *testing.T) {
	isolateConfigMutationRuntime(t)
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(delegatedDecisionConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	read := httptest.NewRecorder()
	ConfigHandler(configPath, false)(read, httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil))
	if read.Code != http.StatusOK {
		t.Fatalf("GET /api/router/config/all = %d: %s", read.Code, read.Body.String())
	}
	requireDelegatedDeployment(t, "config/all", read.Body.Bytes())
	requireTransportHTTPResult(t, UpdateConfigHandler(configPath, false, ""), "/api/router/config/update", read.Body.Bytes())
	saved, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	requireDelegatedDeployment(t, "saved config", saved)
}

func TestSetupKeepsADelegatedDecisionDeployment(t *testing.T) {
	isolateConfigMutationRuntime(t)
	allowLoopbackForTest(t)
	dir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, dir)
	resolver := setupmode.New(configPath, false)
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(delegatedDecisionConfig))
	}))
	defer remote.Close()

	imported := requireTransportHTTPResult(t, SetupImportRemoteHandler(configPath, resolver), "/api/setup/import-remote",
		mustJSONRaw(t, SetupImportRemoteRequest{URL: remote.URL}))
	var importResult SetupImportRemoteResponse
	if err := json.Unmarshal(imported.Body.Bytes(), &importResult); err != nil {
		t.Fatal(err)
	}
	requireDelegatedDeployment(t, "import-remote", importResult.Config)
	validated := requireTransportHTTPResult(t, SetupValidateHandler(configPath, resolver), "/api/setup/validate",
		mustJSONRaw(t, SetupConfigRequest{Config: importResult.Config}))
	var validateResult SetupValidateResponse
	if err := json.Unmarshal(validated.Body.Bytes(), &validateResult); err != nil {
		t.Fatal(err)
	}
	requireDelegatedDeployment(t, "validate", validateResult.Config)
	requireTransportHTTPResult(t, SetupActivateHandler(configPath, false, dir, resolver), "/api/setup/activate",
		mustJSONRaw(t, SetupConfigRequest{Config: validateResult.Config}))
	activated, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	requireDelegatedDeployment(t, "activated config", activated)
}

// The shipped reference config asks a delegated decision deployment.
func TestConfigEditorReadsTheShippedConfig(t *testing.T) {
	configPath, err := filepath.Abs(filepath.Join("..", "..", "..", "config", "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	read := httptest.NewRecorder()
	ConfigHandler(configPath, false)(read, httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil))
	if read.Code != http.StatusOK {
		t.Fatalf("GET /api/router/config/all = %d: %s", read.Code, read.Body.String())
	}
	var document struct {
		Global struct {
			ModelCatalog struct {
				Deployments map[string]routerconfig.ModelDeployment `json:"deployments"`
			} `json:"model_catalog"`
		} `json:"global"`
	}
	if err := json.Unmarshal(read.Body.Bytes(), &document); err != nil {
		t.Fatal(err)
	}
	if auth := document.Global.ModelCatalog.Deployments["decision-delegated"].Auth; auth == nil || auth.Credential != routerconfig.CredentialDelegated {
		t.Fatalf("decision-delegated auth = %+v, want credential %q", auth, routerconfig.CredentialDelegated)
	}
}

// Evaluation records hold the measurements a recipe's model selection reads.
func TestSetupKeepsTheEvaluationSection(t *testing.T) {
	isolateConfigMutationRuntime(t)
	allowLoopbackForTest(t)
	dir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, dir)
	resolver := setupmode.New(configPath, false)
	patch := createValidSetupPatch()
	patch["evaluation"] = map[string]any{"records": []map[string]any{{
		"model":     "test-model",
		"benchmark": "vllm-sr/operator-rating@1.0.0",
		"metrics":   map[string]any{"score": 0.9},
	}}}
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(mustJSONRaw(t, patch))
	}))
	defer remote.Close()
	requireEvaluation := func(surface string, document []byte) {
		t.Helper()
		cfg, err := routerconfig.ParseYAMLBytes(document)
		if err != nil {
			t.Fatalf("%s: Router refuses the config: %v", surface, err)
		}
		if cfg.Evaluation == nil || len(cfg.Evaluation.Records) != 1 {
			t.Fatalf("%s: evaluation = %+v, want the test-model record", surface, cfg.Evaluation)
		}
		if record := cfg.Evaluation.Records[0]; record.Model != "test-model" || record.Benchmark != "vllm-sr/operator-rating@1.0.0" || record.Metrics["score"] != 0.9 {
			t.Fatalf("%s: evaluation record = %+v", surface, record)
		}
	}
	requireEvaluation("remote config", mustJSONRaw(t, patch))

	imported := requireTransportHTTPResult(t, SetupImportRemoteHandler(configPath, resolver), "/api/setup/import-remote",
		mustJSONRaw(t, SetupImportRemoteRequest{URL: remote.URL}))
	var importResult SetupImportRemoteResponse
	if err := json.Unmarshal(imported.Body.Bytes(), &importResult); err != nil {
		t.Fatal(err)
	}
	requireEvaluation("import-remote", importResult.Config)
	validated := requireTransportHTTPResult(t, SetupValidateHandler(configPath, resolver), "/api/setup/validate",
		mustJSONRaw(t, SetupConfigRequest{Config: importResult.Config}))
	var validateResult SetupValidateResponse
	if err := json.Unmarshal(validated.Body.Bytes(), &validateResult); err != nil {
		t.Fatal(err)
	}
	requireEvaluation("validate", validateResult.Config)
	requireTransportHTTPResult(t, SetupActivateHandler(configPath, false, dir, resolver), "/api/setup/activate",
		mustJSONRaw(t, SetupConfigRequest{Config: importResult.Config}))
	activated, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	requireEvaluation("activated config", activated)
}
