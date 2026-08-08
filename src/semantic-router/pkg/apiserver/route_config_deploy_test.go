//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestHandleConfigPatchMergesExistingConfig(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	doc, rawYAML := executeRouterConfigUpdateAndRead(t, http.MethodPatch, configPath, "new_route")
	routing := requireRoutingSection(t, doc)
	requireRoutingProjectionBlock(t, routing, rawYAML)
	requireFirstDecisionName(t, routing, "new_route")
}

func TestHandleConfigPutReplacesExistingConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")

	baseCfg := minimalDeployTestConfig("old_route")
	baseCfg.Projections.Partitions = []config.ProjectionPartition{{
		Name:        "legacy_partition",
		Semantics:   "softmax_exclusive",
		Temperature: 0.1,
		Members:     []string{"math"},
		Default:     "math",
	}}
	baseYAML := mustMarshalCanonicalConfigYAML(t, baseCfg)
	if err := os.WriteFile(configPath, baseYAML, 0o644); err != nil {
		t.Fatalf("write base config: %v", err)
	}

	doc, deployedYAML := executeRouterConfigUpdateAndRead(t, http.MethodPut, configPath, "new_route")
	if _, hasRootDecisions := doc["decisions"]; hasRootDecisions {
		t.Fatalf("expected deploy output to keep decisions under routing, got root-level decisions in:\n%s", string(deployedYAML))
	}

	routing := requireRoutingSection(t, doc)
	requireFirstDecisionName(t, routing, "new_route")
	if projections, ok := routing["projections"].(map[string]any); ok && len(projections) > 0 {
		t.Fatalf("expected replace deploy to drop omitted routing.projections, got %#v", projections)
	}
}

func TestHandleConfigPutPreservesBackendRefNameReturnedByGet(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	if err := os.WriteFile(configPath, mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("old_route")), 0o644); err != nil {
		t.Fatalf("write base config: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: configPath}
	for round := 1; round <= 2; round++ {
		getReq := httptest.NewRequest(http.MethodGet, "/config/router", nil)
		getResp := httptest.NewRecorder()
		apiServer.handleConfigGet(getResp, getReq)
		if getResp.Code != http.StatusOK {
			t.Fatalf("round %d: expected GET 200 OK, got %d: %s", round, getResp.Code, getResp.Body.String())
		}

		var canonicalDocument map[string]any
		if err := json.Unmarshal(getResp.Body.Bytes(), &canonicalDocument); err != nil {
			t.Fatalf("round %d: unmarshal GET response: %v", round, err)
		}
		canonicalYAML, err := yaml.Marshal(canonicalDocument)
		if err != nil {
			t.Fatalf("round %d: marshal GET response as YAML: %v", round, err)
		}
		body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(canonicalYAML)})
		if err != nil {
			t.Fatalf("round %d: marshal PUT request: %v", round, err)
		}

		putReq := httptest.NewRequest(http.MethodPut, "/config/router", bytes.NewReader(body))
		putReq.Header.Set("Content-Type", "application/json")
		putResp := httptest.NewRecorder()
		apiServer.handleConfigPut(putResp, putReq)
		if putResp.Code != http.StatusOK {
			t.Fatalf("round %d: expected PUT 200 OK, got %d: %s", round, putResp.Code, putResp.Body.String())
		}

		updatedYAML, err := os.ReadFile(configPath)
		if err != nil {
			t.Fatalf("round %d: read updated config: %v", round, err)
		}
		var updatedDocument map[string]any
		if err := yaml.Unmarshal(updatedYAML, &updatedDocument); err != nil {
			t.Fatalf("round %d: unmarshal updated config: %v", round, err)
		}
		providers := updatedDocument["providers"].(map[string]any)
		models := providers["models"].([]any)
		model := models[0].(map[string]any)
		backendRefs := model["backend_refs"].([]any)
		backendRef := backendRefs[0].(map[string]any)
		if got, want := backendRef["name"], "qwen-test_primary"; got != want {
			t.Fatalf("round %d: GET then PUT changed backend ref name: got %q, want %q", round, got, want)
		}
	}
}

func TestHandleConfigPutPreservesDisabledRouterReplay(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	payloadCfg := minimalDeployTestConfig("new_route")
	payloadCfg.RouterReplay = config.RouterReplayConfig{
		Enabled:      false,
		StoreBackend: "memory",
		TTLSeconds:   600,
	}
	payloadYAML := mustMarshalCanonicalConfigYAML(t, payloadCfg)
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(payloadYAML)})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: configPath}
	req := httptest.NewRequest(http.MethodPut, "/config/router", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	apiServer.handleConfigPut(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}
	reloaded, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("parse deployed config: %v", err)
	}
	if reloaded.RouterReplay.Enabled {
		t.Fatal("expected router_replay.enabled=false to survive canonical PUT normalization")
	}
}

func TestHandleConfigPutPreservesCanonicalUserDocument(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	payloadYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("new_route"))
	var expected map[string]any
	if err := yaml.Unmarshal(payloadYAML, &expected); err != nil {
		t.Fatalf("decode payload: %v", err)
	}

	providers := expected["providers"].(map[string]any)
	models := providers["models"].([]any)
	model := models[0].(map[string]any)
	backendRefs := model["backend_refs"].([]any)
	backendRefs[0].(map[string]any)["name"] = "vllm_primary"

	routing := expected["routing"].(map[string]any)
	decisions := routing["decisions"].([]any)
	rules := decisions[0].(map[string]any)["rules"].(map[string]any)
	rules["conditions"] = []any{}

	payloadYAML, err := yaml.Marshal(expected)
	if err != nil {
		t.Fatalf("encode payload: %v", err)
	}
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(payloadYAML)})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: configPath}
	req := httptest.NewRequest(http.MethodPut, "/config/router", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	apiServer.handleConfigPut(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	persistedYAML, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read persisted config: %v", err)
	}
	var persisted map[string]any
	if err := yaml.Unmarshal(persistedYAML, &persisted); err != nil {
		t.Fatalf("decode persisted config: %v", err)
	}
	if !reflect.DeepEqual(persisted, expected) {
		t.Fatalf("management PUT rewrote canonical user intent:\nexpected: %#v\nactual: %#v", expected, persisted)
	}
}

func TestHandleConfigPatchRejectsInvalidRemoteEmbeddingProvider(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	invalidCfg := minimalDeployTestConfig("new_route")
	invalidCfg.InlineModels.EmbeddingModels = config.EmbeddingModels{
		EmbeddingConfig: config.HNSWConfig{
			Backend:         config.EmbeddingBackendOpenAICompatible,
			ModelType:       config.EmbeddingModelTypeRemote,
			TargetDimension: 768,
		},
		Endpoint: config.EmbeddingEndpointConfig{
			BaseURL:    "embedding-service:8000/v1",
			Dimensions: 1536,
		},
	}
	payloadYAML := mustMarshalCanonicalConfigYAML(t, invalidCfg)
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(payloadYAML)})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: configPath}
	req := httptest.NewRequest(http.MethodPatch, "/config/router", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	apiServer.handleConfigPatch(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 Bad Request, got %d: %s", rr.Code, rr.Body.String())
	}
	response := rr.Body.String()
	for _, want := range []string{"CONFIG_VALIDATION_ERROR", "endpoint.base_url", "endpoint.model", "endpoint.dimensions"} {
		if !strings.Contains(response, want) {
			t.Fatalf("expected response to contain %q, got: %s", want, response)
		}
	}
}

func writeDeployTestBaseConfig(t *testing.T) string {
	t.Helper()

	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	baseCfg := minimalDeployTestConfig("old_route")
	baseCfg.Projections.Partitions = []config.ProjectionPartition{{
		Name:        "legacy_partition",
		Semantics:   "softmax_exclusive",
		Temperature: 0.1,
		Members:     []string{"math"},
		Default:     "math",
	}}
	baseYAML := mustMarshalCanonicalConfigYAML(t, baseCfg)
	if err := os.WriteFile(configPath, baseYAML, 0o644); err != nil {
		t.Fatalf("write base config: %v", err)
	}
	return configPath
}

func executeRouterConfigUpdateAndRead(
	t *testing.T,
	method string,
	configPath string,
	decisionName string,
) (map[string]any, []byte) {
	t.Helper()

	payloadYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig(decisionName))
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(payloadYAML)})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: configPath}
	req := httptest.NewRequest(method, "/config/router", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	switch method {
	case http.MethodPatch:
		apiServer.handleConfigPatch(rr, req)
	case http.MethodPut:
		apiServer.handleConfigPut(rr, req)
	default:
		t.Fatalf("unsupported update method %q", method)
	}

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}

	var doc map[string]any
	if err := yaml.Unmarshal(data, &doc); err != nil {
		t.Fatalf("yaml.Unmarshal updated config: %v", err)
	}

	return doc, data
}

func requireRoutingSection(t *testing.T, doc map[string]any) map[string]any {
	t.Helper()

	routing, ok := doc["routing"].(map[string]any)
	if !ok {
		t.Fatalf("expected routing block in updated config, got %#v", doc)
	}
	return routing
}

func requireRoutingProjectionBlock(t *testing.T, routing map[string]any, rawYAML []byte) {
	t.Helper()

	projections, ok := routing["projections"].(map[string]any)
	if !ok || len(projections) == 0 {
		t.Fatalf("expected merge to preserve existing routing.projections, got %#v in:\n%s", routing["projections"], string(rawYAML))
	}
}

func requireFirstDecisionName(t *testing.T, routing map[string]any, want string) {
	t.Helper()

	decisions, ok := routing["decisions"].([]any)
	if !ok || len(decisions) == 0 {
		t.Fatalf("expected routing.decisions in updated config, got %#v", routing["decisions"])
	}
	firstDecision, ok := decisions[0].(map[string]any)
	if !ok {
		t.Fatalf("expected routing.decisions[0] to be a map, got %#v", decisions[0])
	}
	if got := firstDecision["name"]; got != want {
		t.Fatalf("expected routing.decisions[0].name = %s, got %v", want, got)
	}
}

func minimalDeployTestConfig(decisionName string) *config.RouterConfig {
	return &config.RouterConfig{
		APIServer: config.APIServer{
			Listeners: []config.Listener{{
				Name:    "http-8899",
				Address: "0.0.0.0",
				Port:    8899,
				Timeout: "120s",
			}},
		},
		BackendModels: config.BackendModels{
			DefaultModel: "qwen-test",
			ModelConfig: map[string]config.ModelParams{
				"qwen-test": {},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name:     "primary",
				Address:  "vllm",
				Port:     8000,
				Protocol: "http",
				Model:    "qwen-test",
				Weight:   1,
			}},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				Categories: []config.Category{{
					CategoryMetadata: config.CategoryMetadata{Name: "math"},
				}},
			},
			Decisions: []config.Decision{{
				Name:     decisionName,
				Priority: 100,
				Tier:     1,
				Rules: config.RuleCombination{
					Operator: "AND",
					Conditions: []config.RuleNode{{
						Type: "domain",
						Name: "math",
					}},
				},
				ModelRefs: []config.ModelRef{{
					Model: "qwen-test",
				}},
			}},
		},
	}
}

func mustMarshalCanonicalConfigYAML(t *testing.T, cfg *config.RouterConfig) []byte {
	t.Helper()
	canonical := config.CanonicalConfigFromRouterConfig(cfg)
	data, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatalf("yaml.Marshal canonical config: %v", err)
	}
	return data
}

func rollbackErrorCode(t *testing.T, body []byte) string {
	t.Helper()

	var resp struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("unmarshal error response %q: %v", string(body), err)
	}
	return resp.Error.Code
}

func postRollback(t *testing.T, configPath, version string) *httptest.ResponseRecorder {
	t.Helper()

	body, err := json.Marshal(map[string]string{"version": version})
	if err != nil {
		t.Fatalf("marshal rollback body: %v", err)
	}
	apiServer := &ClassificationAPIServer{configPath: configPath}
	req := httptest.NewRequest(http.MethodPost, "/config/router/rollback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	apiServer.handleConfigRollback(rr, req)
	return rr
}

// TestHandleConfigRollbackRejectsMaliciousVersion ensures the rollback endpoint
// rejects any version that is not a canonical backup timestamp before it is used
// to build a filesystem path, closing the path-traversal vector.
func TestHandleConfigRollbackRejectsMaliciousVersion(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	originalConfig, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read base config: %v", err)
	}

	cases := []struct {
		name    string
		version string
	}{
		{"parent traversal", "../../../../etc/passwd"},
		{"deep traversal", "../../../../../../../../etc/hostname"},
		{"url encoded traversal", "..%2f..%2fetc%2fpasswd"},
		{"embedded traversal", "foo/../../bar"},
		{"absolute path", "/etc/passwd"},
		{"bare word", "config"},
		{"valid prefix yaml suffix", "20240101-000000.yaml"},
		{"valid prefix then traversal", "20240101-000000/../../secret"},
		{"wrong separator", "20240101_000000"},
		{"too short date", "2024010-000000"},
		{"non numeric", "abcdefgh-ijklmn"},
		{"shell metachars", "; rm -rf /"},
		{"dot dot", ".."},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rr := postRollback(t, configPath, tc.version)

			if rr.Code != http.StatusBadRequest {
				t.Fatalf("version %q: expected 400, got %d: %s", tc.version, rr.Code, rr.Body.String())
			}
			if code := rollbackErrorCode(t, rr.Body.Bytes()); code != "INVALID_INPUT" {
				t.Fatalf("version %q: expected error code INVALID_INPUT, got %q", tc.version, code)
			}
		})
	}

	// A rejected rollback must never mutate the active config.
	afterConfig, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("re-read base config: %v", err)
	}
	if !bytes.Equal(originalConfig, afterConfig) {
		t.Fatal("rejected rollback mutated the active config")
	}
}

// TestHandleConfigRollbackAcceptsWellFormedVersion proves the allowlist does not
// over-reject the canonical YYYYMMDD-HHMMSS format: a correctly-formatted version
// with no backup file passes validation and fails later at the lookup (404).
func TestHandleConfigRollbackAcceptsWellFormedVersion(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)

	rr := postRollback(t, configPath, "20240101-000000")

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for well-formed missing version, got %d: %s", rr.Code, rr.Body.String())
	}
	if code := rollbackErrorCode(t, rr.Body.Bytes()); code != "VERSION_NOT_FOUND" {
		t.Fatalf("expected error code VERSION_NOT_FOUND, got %q", code)
	}
}

func TestHandleConfigRollbackAcceptsCollisionSuffixedVersion(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)

	rr := postRollback(t, configPath, "20240101-000000-001")

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for well-formed missing suffixed version, got %d: %s", rr.Code, rr.Body.String())
	}
	if code := rollbackErrorCode(t, rr.Body.Bytes()); code != "VERSION_NOT_FOUND" {
		t.Fatalf("expected error code VERSION_NOT_FOUND, got %q", code)
	}
}

func TestNextConfigVersionAvoidsSameSecondBackupCollisions(t *testing.T) {
	backupDir := t.TempDir()
	now := time.Date(2026, time.August, 2, 10, 11, 12, 0, time.UTC)
	base := nextConfigVersion(backupDir, now)
	if base != "20260802-101112" {
		t.Fatalf("base version = %q", base)
	}
	if err := os.WriteFile(filepath.Join(backupDir, "config."+base+".yaml"), []byte("first"), 0o644); err != nil {
		t.Fatalf("write first backup: %v", err)
	}

	second := nextConfigVersion(backupDir, now)
	if second != "20260802-101112-001" {
		t.Fatalf("second version = %q", second)
	}
	if err := os.WriteFile(filepath.Join(backupDir, "config."+second+".yaml"), []byte("second"), 0o644); err != nil {
		t.Fatalf("write second backup: %v", err)
	}
	if third := nextConfigVersion(backupDir, now); third != "20260802-101112-002" {
		t.Fatalf("third version = %q", third)
	}
}

func TestHandleConfigRollbackRejectsIncompatibleLocalClassifier(t *testing.T) {
	configDir := t.TempDir()
	configPath := filepath.Join(configDir, "config.yaml")
	current := []byte(localClassifierReloadConfig("models/risk-v1"))
	if err := os.WriteFile(configPath, current, 0o600); err != nil {
		t.Fatalf("write current config: %v", err)
	}
	backupDir := filepath.Join(configDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o700); err != nil {
		t.Fatalf("create backup dir: %v", err)
	}
	version := "20240101-000000"
	backupPath := filepath.Join(backupDir, "config."+version+".yaml")
	if err := os.WriteFile(
		backupPath,
		[]byte(localClassifierReloadConfig("models/risk-v2")),
		0o600,
	); err != nil {
		t.Fatalf("write backup config: %v", err)
	}

	response := postRollback(t, configPath, version)

	if response.Code != http.StatusConflict {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if code := rollbackErrorCode(t, response.Body.Bytes()); code != "RESTART_REQUIRED" {
		t.Fatalf("error code = %q, want RESTART_REQUIRED", code)
	}
	after, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read current config: %v", err)
	}
	if !bytes.Equal(after, current) {
		t.Fatal("incompatible rollback mutated the active config")
	}
}

func TestHandleConfigHashReturnsHashAndNoPath(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	activeCfg, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("parse active config: %v", err)
	}
	apiServer := &ClassificationAPIServer{configPath: configPath, config: activeCfg}

	req := httptest.NewRequest(http.MethodGet, "/config/hash", nil)
	rr := httptest.NewRecorder()
	apiServer.handleConfigHash(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp configHashResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json.Unmarshal response: %v", err)
	}

	if len(resp.Hash) != 64 || len(resp.RuntimeHash) != 64 {
		t.Fatalf("expected source and runtime hashes, got %+v", resp)
	}
	if resp.Status != "active" {
		t.Fatalf("expected active status, got %+v", resp)
	}
	var raw map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatalf("json.Unmarshal raw response: %v", err)
	}
	if _, hasPath := raw["path"]; hasPath {
		t.Error("response should not expose filesystem path")
	}
}

func TestHandleConfigHashErrorsWithoutConfigPath(t *testing.T) {
	apiServer := &ClassificationAPIServer{configPath: ""}

	req := httptest.NewRequest(http.MethodGet, "/config/hash", nil)
	rr := httptest.NewRecorder()
	apiServer.handleConfigHash(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHandleConfigHashStableForSameContent(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	apiServer := &ClassificationAPIServer{configPath: configPath}

	hashes := make([]string, 2)
	for i := range hashes {
		req := httptest.NewRequest(http.MethodGet, "/config/hash", nil)
		rr := httptest.NewRecorder()
		apiServer.handleConfigHash(rr, req)

		var resp configHashResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}
		hashes[i] = resp.Hash
	}

	if hashes[0] != hashes[1] {
		t.Fatalf("expected identical hashes for unchanged config, got %q vs %q", hashes[0], hashes[1])
	}
}

func TestHandleConfigHashReportsPendingRuntime(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	registry := routerruntime.NewRegistry(&config.RouterConfig{DocumentHash: "previous-runtime"})
	apiServer := &ClassificationAPIServer{configPath: configPath, runtimeRegistry: registry}

	req := httptest.NewRequest(http.MethodGet, "/config/hash", nil)
	rr := httptest.NewRecorder()
	apiServer.handleConfigHash(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp configHashResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("json.Unmarshal response: %v", err)
	}
	if resp.Status != "pending" || resp.ActiveHash != "previous-runtime" {
		t.Fatalf("expected pending runtime status, got %+v", resp)
	}
}

func TestWaitForRuntimeConfigActivationRecognizesPublishedDocument(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	activeCfg, err := config.Parse(configPath)
	if err != nil {
		t.Fatalf("parse active config: %v", err)
	}
	registry := routerruntime.NewRegistry(activeCfg)
	apiServer := &ClassificationAPIServer{configPath: configPath, runtimeRegistry: registry}

	runtimeHash, status := apiServer.waitForRuntimeConfigActivation(configPath)
	if status != "active" || runtimeHash != activeCfg.DocumentHash {
		t.Fatalf("activation result = (%q, %q), want (%q, active)", runtimeHash, status, activeCfg.DocumentHash)
	}
}

func TestConfigVersionSourceMetadataDistinguishesMutationOrigins(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	server := &ClassificationAPIServer{}

	apiVersion, backupDir := server.recordRouterConfigArtifacts(
		configPath, []byte("version: v0.3\n"), "",
	)
	if got := readConfigVersionSource(backupDir, apiVersion); got != configVersionSourceAPI {
		t.Fatalf("API backup source = %q, want %q", got, configVersionSourceAPI)
	}

	dslVersion, _ := server.recordRouterConfigArtifacts(
		configPath, []byte("version: v0.3\n"), "ROUTE default {}",
	)
	if got := readConfigVersionSource(backupDir, dslVersion); got != configVersionSourceDSL {
		t.Fatalf("DSL backup source = %q, want %q", got, configVersionSourceDSL)
	}

	legacyVersion := "20260802-120000"
	if got := readConfigVersionSource(backupDir, legacyVersion); got != configVersionSourceLegacy {
		t.Fatalf("backup without metadata source = %q, want %q", got, configVersionSourceLegacy)
	}
}

func TestConfigCleanupBackupsRemovesSourceSidecar(t *testing.T) {
	backupDir := t.TempDir()
	for i := 0; i <= maxBackups; i++ {
		version := fmt.Sprintf("20260802-1200%02d", i)
		if err := os.WriteFile(
			filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version)),
			[]byte("version: v0.3\n"),
			0o644,
		); err != nil {
			t.Fatalf("write backup: %v", err)
		}
		writeConfigVersionSource(backupDir, version, configVersionSourceAPI)
	}

	configCleanupBackups(backupDir)
	oldest := "20260802-120000"
	if _, err := os.Stat(configVersionSourcePath(backupDir, oldest)); !os.IsNotExist(err) {
		t.Fatalf("oldest source sidecar should be removed, stat err=%v", err)
	}
}
