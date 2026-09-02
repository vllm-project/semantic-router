package router

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

func TestEvaluationServiceEndpointConversionKeepsOnlySecretReference(t *testing.T) {
	if endpoint := evaluationServiceEndpoint(config.EvaluationServiceEndpointConfig{}); endpoint != nil {
		t.Fatalf("zero endpoint = %+v, want nil", endpoint)
	}
	endpoint := evaluationServiceEndpoint(config.EvaluationServiceEndpointConfig{
		URL: "https://ledger.internal", APIKeyEnv: "LEDGER_TOKEN", Timeout: 45 * time.Second,
	})
	if endpoint == nil || endpoint.URL != "https://ledger.internal" || endpoint.TimeoutSeconds != 45 ||
		endpoint.APIKey == nil || endpoint.APIKey.Env != "LEDGER_TOKEN" {
		t.Fatalf("converted endpoint = %+v", endpoint)
	}
	encoded, err := json.Marshal(endpoint)
	if err != nil || strings.Contains(string(encoded), "secret-value") {
		t.Fatalf("endpoint serialized secret material: %s err=%v", encoded, err)
	}
}

func TestEvaluationRoutesEnableAuthenticatedRoutingAndConfiguredLedgers(t *testing.T) {
	root := t.TempDir()
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte(`version: v0.3
global:
  router:
    auto_model_names: [test-mom]
providers:
  defaults:
    default_model: model-fast
  models:
    - name: model-fast
      backend_refs: [{endpoint: fast.models.test:8000}]
    - name: model-strong
      backend_refs: [{endpoint: strong.models.test:8000}]
routing:
  modelCards:
    - {name: model-fast, modality: text}
    - {name: model-strong, modality: text}
  decisions:
    - name: route
      rules: {}
      modelRefs: [{model: model-fast}, {model: model-strong}]
`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(recipe.ManagementCredentialEnv, "")
	store := recipe.NewStore(recipe.StoreOptions{Root: filepath.Join(root, "recipes"), ConfigPath: configPath})
	if _, err := store.EnsureManagementCredential(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ROUTER_EVAL_TOKEN", "dedicated-router-evaluation-token")
	t.Setenv("VLLM_SR_SOURCE_REVISION", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	cfg := &config.Config{
		EvaluationEnabled: true, EvaluationDataDir: filepath.Join(root, "evaluation"), PythonPath: "python3",
		AbsConfigPath: configPath, RouterAPIURL: "https://router.internal", EnvoyURL: "https://envoy.internal",
		EvaluationRouterAPIKeyEnv: "ROUTER_EVAL_TOKEN",
		EvaluationAgentTaskLedger: config.EvaluationServiceEndpointConfig{
			URL: "https://agent-task.internal", APIKeyEnv: "AGENT_TASK_TOKEN", Timeout: 10 * time.Second,
		},
		EvaluationFaultRecoveryLedger: config.EvaluationServiceEndpointConfig{
			URL: "https://fault.internal", APIKeyEnv: "FAULT_TOKEN", Timeout: 15 * time.Second,
		},
		EvaluationHardPolicyLedger: config.EvaluationServiceEndpointConfig{
			URL: "https://policy.internal", APIKeyEnv: "POLICY_TOKEN", Timeout: 20 * time.Second,
		},
		EvaluationProductionExperimentLedger: config.EvaluationServiceEndpointConfig{
			URL: "https://experiment.internal", APIKeyEnv: "EXPERIMENT_TOKEN", Timeout: 25 * time.Second,
		},
	}
	mux := http.NewServeMux()
	service := registerEvaluationRoutes(mux, cfg, store)
	if service == nil || !cfg.EvaluationAvailable {
		t.Fatalf("authenticated Evaluation unavailable: reason=%q", cfg.EvaluationUnavailableReason)
	}
	t.Cleanup(func() { _ = service.Close() })

	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/evaluation/v1/catalog", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("catalog status=%d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.Bytes()
	var catalog evaluationplane.Catalog
	if err := json.NewDecoder(bytes.NewReader(body)).Decode(&catalog); err != nil {
		t.Fatal(err)
	}
	var target *evaluationplane.CatalogTarget
	for index := range catalog.Targets {
		if catalog.Targets[index].Kind == "mixture-of-models" {
			target = &catalog.Targets[index]
			break
		}
	}
	wantTracks := []evaluationplane.TrackID{"routing", "model_pool", "joint", "agentic", "preference", "safety", "capacity"}
	if target == nil || !reflect.DeepEqual(target.TrackIDs, wantTracks) ||
		target.Labels["router_auth"] != "dedicated-evaluation-credential-configured" {
		t.Fatalf("configured target = %+v", target)
	}
	agentTaskConfigured := false
	for _, suite := range catalog.Suites {
		if suite.ID == "live-agent-tasks" && len(suite.Methods) == 1 &&
			suite.Methods[0].Status == "configured" && len(suite.Methods[0].QualifiedGateIDs) == 0 {
			agentTaskConfigured = true
		}
	}
	if !agentTaskConfigured {
		t.Fatalf("dedicated agent-task method was not configured: %+v", catalog.Suites)
	}
	for _, forbidden := range []string{
		"dedicated-router-evaluation-token", "ROUTER_EVAL_TOKEN", "AGENT_TASK_TOKEN", "FAULT_TOKEN", "POLICY_TOKEN", "EXPERIMENT_TOKEN",
		"router.internal", "agent-task.internal", "fault.internal", "policy.internal", "experiment.internal",
	} {
		if strings.Contains(string(body), forbidden) {
			t.Fatalf("catalog leaked server-owned Evaluation config %q: %s", forbidden, body)
		}
	}
}

func TestEvaluationRoutesLoadDeploymentScopedTargetsWithoutCatalogLeaks(t *testing.T) {
	root := t.TempDir()
	configPath := filepath.Join(root, "config.yaml")
	configBytes := []byte(`version: v0.3
global:
  router:
    auto_model_names: [test-mom]
providers:
  defaults: {default_model: model-fast}
  models:
    - name: model-fast
      backend_refs: [{endpoint: fast.models.test:8000}]
    - name: model-strong
      backend_refs: [{endpoint: strong.models.test:8000}]
routing:
  modelCards:
    - {name: model-fast, modality: text}
    - {name: model-strong, modality: text}
  decisions:
    - name: route
      rules: {}
      modelRefs: [{model: model-fast}, {model: model-strong}]
`)
	if err := os.WriteFile(configPath, configBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	deployments := filepath.Join(root, "deployments")
	if err := os.Mkdir(deployments, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"baseline.yaml", "candidate.yaml"} {
		if err := os.WriteFile(filepath.Join(deployments, name), configBytes, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	registry := `{"schema_version":"evaluation-deployments.v1","deployments":[` +
		`{"id":"baseline","name":"Baseline","config_file":"baseline.yaml","router_origin":"https://baseline-router.private","envoy_origin":"https://baseline-envoy.private"},` +
		`{"id":"candidate","name":"Candidate","config_file":"candidate.yaml","router_origin":"https://candidate-router.private","envoy_origin":"https://candidate-envoy.private"}]}`
	if err := os.WriteFile(filepath.Join(deployments, "registry.json"), []byte(registry), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_SOURCE_REVISION", strings.Repeat("a", 40))
	cfg := &config.Config{
		EvaluationEnabled: true, EvaluationDataDir: filepath.Join(root, "evaluation"),
		EvaluationDeploymentsDir: deployments, PythonPath: "python3", AbsConfigPath: configPath,
	}
	mux := http.NewServeMux()
	service := registerEvaluationRoutes(mux, cfg)
	if service == nil || !cfg.EvaluationAvailable {
		t.Fatalf("deployment Evaluation unavailable: %q", cfg.EvaluationUnavailableReason)
	}
	t.Cleanup(func() { _ = service.Close() })
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/evaluation/v1/catalog", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("catalog status=%d body=%s", response.Code, response.Body.String())
	}
	var catalog evaluationplane.Catalog
	if err := json.Unmarshal(response.Body.Bytes(), &catalog); err != nil {
		t.Fatal(err)
	}
	deploymentsSeen := map[string]bool{}
	for _, target := range catalog.Targets {
		if target.Kind == "mixture-of-models" {
			deploymentsSeen[target.Labels["deployment"]] = true
		}
	}
	if !deploymentsSeen["Baseline"] || !deploymentsSeen["Candidate"] {
		t.Fatalf("deployment targets missing: %+v", catalog.Targets)
	}
	for _, forbidden := range []string{"baseline-router.private", "candidate-envoy.private", "baseline.yaml", deployments} {
		if strings.Contains(response.Body.String(), forbidden) {
			t.Fatalf("catalog leaked deployment value %q", forbidden)
		}
	}
}
