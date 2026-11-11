package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The Router loads the earlier name of the systemone signal and selector,
// decision, as systemone.
const renamedDecisionFormsConfig = `version: v0.5
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
    - name: second-model
      provider_model_id: second-model
      backend_refs:
        - name: secondary
          provider: vllm
          endpoint: host.docker.internal:8001
          protocol: http
recipes:
  - name: default
    signals:
      decision:
        - name: support_team
          description: Which team should handle the request.
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
            - type: decision
              name: support_team
              label: billing
        modelRefs:
          - model: test-model
          - model: second-model
        algorithm:
          type: decision
          decision:
            instructions: Which model should answer this billing request?
`

// requireSystemOneForms checks that a config the Dashboard returned or saved
// names the signal and the selector systemone, and that the Router reads it.
func requireSystemOneForms(t *testing.T, surface string, document []byte) {
	t.Helper()
	var shown struct {
		Recipes []struct {
			Signals struct {
				SystemOne []struct {
					Name string `yaml:"name"`
				} `yaml:"systemone"`
			} `yaml:"signals"`
			Decisions []struct {
				Algorithm struct {
					Type string `yaml:"type"`
				} `yaml:"algorithm"`
			} `yaml:"decisions"`
		} `yaml:"recipes"`
	}
	if err := yaml.Unmarshal(document, &shown); err != nil {
		t.Fatal(err)
	}
	if len(shown.Recipes) != 1 || len(shown.Recipes[0].Signals.SystemOne) != 1 || shown.Recipes[0].Signals.SystemOne[0].Name != "support_team" {
		t.Fatalf("%s: recipes = %+v, want the support_team systemone signal", surface, shown.Recipes)
	}
	if decisions := shown.Recipes[0].Decisions; len(decisions) != 1 || decisions[0].Algorithm.Type != routerconfig.DecisionAlgorithmSystemOne {
		t.Fatalf("%s: decisions = %+v, want the systemone selector", surface, decisions)
	}
	cfg, err := routerconfig.ParseYAMLBytes(document)
	if err != nil {
		t.Fatalf("%s: Router refuses the config: %v", surface, err)
	}
	if len(cfg.SystemOneRules) != 1 || cfg.SystemOneRules[0].Name != "support_team" {
		t.Fatalf("%s: Router systemone rules = %+v", surface, cfg.SystemOneRules)
	}
}

func TestConfigEditorReadsTheRenamedDecisionForms(t *testing.T) {
	isolateConfigMutationRuntime(t)
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(renamedDecisionFormsConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	read := httptest.NewRecorder()
	ConfigHandler(configPath, false)(read, httptest.NewRequest(http.MethodGet, "/api/router/config/all", nil))
	if read.Code != http.StatusOK {
		t.Fatalf("GET /api/router/config/all = %d: %s", read.Code, read.Body.String())
	}
	requireSystemOneForms(t, "config/all", read.Body.Bytes())
	requireTransportHTTPResult(t, UpdateConfigHandler(configPath, false, ""), "/api/router/config/update", read.Body.Bytes())
	saved, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	requireSystemOneForms(t, "saved config", saved)
}

func TestSetupImportReadsTheRenamedDecisionForms(t *testing.T) {
	isolateConfigMutationRuntime(t)
	allowLoopbackForTest(t)
	dir := t.TempDir()
	configPath := createBootstrapSetupConfig(t, dir)
	resolver := setupmode.New(configPath, false)
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(renamedDecisionFormsConfig))
	}))
	defer remote.Close()

	imported := requireTransportHTTPResult(t, SetupImportRemoteHandler(configPath, resolver), "/api/setup/import-remote",
		mustJSONRaw(t, SetupImportRemoteRequest{URL: remote.URL}))
	var importResult SetupImportRemoteResponse
	if err := json.Unmarshal(imported.Body.Bytes(), &importResult); err != nil {
		t.Fatal(err)
	}
	if importResult.Signals != 1 {
		t.Fatalf("import summary signals = %d, want 1", importResult.Signals)
	}
	requireSystemOneForms(t, "import-remote", importResult.Config)
	requireTransportHTTPResult(t, SetupActivateHandler(configPath, false, dir, resolver), "/api/setup/activate",
		mustJSONRaw(t, SetupConfigRequest{Config: importResult.Config}))
	activated, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	requireSystemOneForms(t, "activated config", activated)
}

// The Builder's deploy preview shows the current config as the Router reads it.
func TestDeployPreviewReadsTheRenamedDecisionForms(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte(renamedDecisionFormsConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	body := mustJSONRaw(t, DeployRequest{YAML: "recipes: []", Mode: "merge"})
	preview := httptest.NewRecorder()
	DeployPreviewHandler(configPath, false)(preview,
		httptest.NewRequest(http.MethodPost, "/api/router/config/deploy/preview", bytes.NewReader(body)))
	if preview.Code != http.StatusOK {
		t.Fatalf("deploy preview = %d: %s", preview.Code, preview.Body.String())
	}
	var result DeployPreviewResponse
	if err := json.Unmarshal(preview.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	requireSystemOneForms(t, "preview current", []byte(result.Current))
}
