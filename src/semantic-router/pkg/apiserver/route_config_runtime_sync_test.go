//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	routerdsl "github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

func TestResolveConfigPersistencePathsUsesRuntimeOverrideSourcePath(t *testing.T) {
	t.Setenv(sourceConfigPathEnv, "/app/config.yaml")
	t.Setenv(runtimeConfigPathEnv, "/app/.vllm-sr/runtime-config.yaml")

	paths := resolveConfigPersistencePaths("/app/.vllm-sr/runtime-config.yaml")

	if paths.sourcePath != "/app/config.yaml" {
		t.Fatalf("sourcePath = %q, want /app/config.yaml", paths.sourcePath)
	}
	if paths.runtimePath != "/app/.vllm-sr/runtime-config.yaml" {
		t.Fatalf("runtimePath = %q, want /app/.vllm-sr/runtime-config.yaml", paths.runtimePath)
	}
	if !paths.usesRuntimeOverride() {
		t.Fatal("expected runtime override to be detected")
	}
}

func TestHandleConfigPutWritesSourceConfigAndSyncsRuntimeOverride(t *testing.T) {
	tempDir, sourcePath, runtimePath := setupRuntimeOverrideConfigFiles(t, "old_route", "old_route")
	deployYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("new_route"))
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(deployYAML), DSL: "ROUTE new_route"})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: runtimePath}
	req := httptest.NewRequest(http.MethodPut, "/config/router", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	apiServer.handleConfigPut(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	assertDeployedDecisionName(t, sourcePath, "new_route")
	assertDeployedDecisionName(t, runtimePath, "new_route")

	backupDir := filepath.Join(tempDir, ".vllm-sr", "config-backups")
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		t.Fatalf("read backup dir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 backup in %s, got %d", backupDir, len(entries))
	}

	dslPath := filepath.Join(tempDir, ".vllm-sr", "config.dsl")
	if _, err := os.Stat(dslPath); err != nil {
		t.Fatalf("expected archived DSL at %s: %v", dslPath, err)
	}

	nestedBackupDir := filepath.Join(tempDir, ".vllm-sr", ".vllm-sr", "config-backups")
	if _, err := os.Stat(nestedBackupDir); !os.IsNotExist(err) {
		t.Fatalf("did not expect nested backup dir %s to exist", nestedBackupDir)
	}
}

func TestHandleConfigPutArchivesDecisionTreeDSLButConfigGetStaysCanonical(t *testing.T) {
	tempDir, sourcePath, runtimePath := setupRuntimeOverrideConfigFiles(t, "old_route", "old_route")
	decisionTreeDSL := strings.TrimSpace(`
SIGNAL domain math { mmlu_categories: ["math"] }

DECISION_TREE routing_policy {
  IF domain("math") {
    NAME "math_route"
    MODEL "qwen-test"
  }
  ELSE {
    NAME "fallback_route"
    MODEL "qwen-test"
  }
}
`)
	deployYAML := mustMarshalDecisionTreeDeployYAML(t, decisionTreeDSL)
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(deployYAML), DSL: decisionTreeDSL})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: runtimePath}
	req := httptest.NewRequest(http.MethodPut, "/config/router", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	apiServer.handleConfigPut(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	assertDeployedDecisionNames(t, sourcePath, "math_route", "fallback_route")
	assertDeployedDecisionNames(t, runtimePath, "math_route", "fallback_route")
	assertArchivedDecisionTreeDSL(t, tempDir, decisionTreeDSL)
	assertConfigGetStaysCanonical(t, apiServer, "math_route", "fallback_route")
}

func assertArchivedDecisionTreeDSL(t *testing.T, tempDir string, want string) {
	t.Helper()

	dslPath := filepath.Join(tempDir, ".vllm-sr", "config.dsl")
	archived, err := os.ReadFile(dslPath)
	if err != nil {
		t.Fatalf("expected archived DSL at %s: %v", dslPath, err)
	}
	if got := strings.TrimSpace(string(archived)); got != want {
		t.Fatalf("archived DSL mismatch\nwant:\n%s\n\ngot:\n%s", want, got)
	}
	if !strings.Contains(string(archived), "DECISION_TREE routing_policy") {
		t.Fatalf("expected archived DSL to preserve DECISION_TREE authoring, got:\n%s", string(archived))
	}
}

func assertConfigGetStaysCanonical(
	t *testing.T,
	apiServer *ClassificationAPIServer,
	wantDecisions ...string,
) {
	t.Helper()

	bodyText, doc := readRouterConfigDocument(t, apiServer)
	assertCanonicalRouterConfigResponse(t, bodyText)
	assertCanonicalRouterConfigDecisions(t, doc, wantDecisions...)
}

func readRouterConfigDocument(
	t *testing.T, apiServer *ClassificationAPIServer,
) (string, map[string]any) {
	t.Helper()

	getReq := httptest.NewRequest(http.MethodGet, "/config/router", nil)
	getRR := httptest.NewRecorder()
	apiServer.handleConfigGet(getRR, getReq)
	if getRR.Code != http.StatusOK {
		t.Fatalf("expected GET /config/router to return 200 OK, got %d: %s", getRR.Code, getRR.Body.String())
	}

	bodyText := getRR.Body.String()
	var doc map[string]any
	if err := json.Unmarshal(getRR.Body.Bytes(), &doc); err != nil {
		t.Fatalf("json.Unmarshal GET /config/router: %v", err)
	}
	return bodyText, doc
}

func assertCanonicalRouterConfigResponse(t *testing.T, bodyText string) {
	t.Helper()

	if strings.Contains(bodyText, "DECISION_TREE") || strings.Contains(bodyText, "ELSE IF") {
		t.Fatalf("expected /config/router to stay on canonical flat config, got:\n%s", bodyText)
	}
}

func assertCanonicalRouterConfigDecisions(
	t *testing.T,
	doc map[string]any,
	wantDecisions ...string,
) {
	t.Helper()

	routing, ok := doc["routing"].(map[string]any)
	if !ok {
		t.Fatalf("expected routing object in GET /config/router response, got %#v", doc["routing"])
	}
	decisions, ok := routing["decisions"].([]any)
	if !ok {
		t.Fatalf("expected routing.decisions array in GET /config/router response, got %#v", routing["decisions"])
	}
	if len(decisions) != 2 {
		t.Fatalf("expected 2 flattened routing.decisions entries, got %d", len(decisions))
	}
	for index, wantDecision := range wantDecisions {
		requireNamedDecision(t, decisions[index], wantDecision)
	}
}

func TestHandleConfigRollbackReadsSourceBackupDirAndSyncsRuntimeOverride(t *testing.T) {
	tempDir := t.TempDir()
	sourcePath := filepath.Join(tempDir, "config.yaml")
	runtimePath := filepath.Join(tempDir, ".vllm-sr", "runtime-config.yaml")
	backupDir := filepath.Join(tempDir, ".vllm-sr", "config-backups")
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		t.Fatalf("mkdir backup dir: %v", err)
	}

	currentYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("current_route"))
	rollbackYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("rolled_back_route"))
	if err := os.WriteFile(sourcePath, currentYAML, 0o644); err != nil {
		t.Fatalf("write source config: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(runtimePath), 0o755); err != nil {
		t.Fatalf("mkdir runtime dir: %v", err)
	}
	if err := os.WriteFile(runtimePath, currentYAML, 0o644); err != nil {
		t.Fatalf("write runtime config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(backupDir, "config.20260323-120000.yaml"), rollbackYAML, 0o644); err != nil {
		t.Fatalf("write backup config: %v", err)
	}

	t.Setenv(sourceConfigPathEnv, sourcePath)
	t.Setenv(runtimeConfigPathEnv, runtimePath)

	previousRunner := runtimeConfigSyncRunner
	runtimeConfigSyncRunner = func(configPath string) (string, error) {
		data, err := os.ReadFile(configPath)
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(runtimePath, data, 0o644); err != nil {
			return "", err
		}
		return runtimePath, nil
	}
	defer func() { runtimeConfigSyncRunner = previousRunner }()

	body := []byte(`{"version":"20260323-120000"}`)
	apiServer := &ClassificationAPIServer{configPath: runtimePath}
	req := httptest.NewRequest(http.MethodPost, "/config/router/rollback", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	apiServer.handleConfigRollback(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	assertDeployedDecisionName(t, sourcePath, "rolled_back_route")
	assertDeployedDecisionName(t, runtimePath, "rolled_back_route")
}

func assertDeployedDecisionName(t *testing.T, configPath string, want string) {
	t.Helper()

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config %s: %v", configPath, err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse config %s: %v", configPath, err)
	}
	if len(cfg.Decisions) == 0 {
		t.Fatalf("expected at least one decision in %s", configPath)
	}
	if got := cfg.Decisions[0].Name; got != want {
		t.Fatalf("decision name in %s = %q, want %q", configPath, got, want)
	}
}

func assertDeployedDecisionNames(t *testing.T, configPath string, want ...string) {
	t.Helper()

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config %s: %v", configPath, err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse config %s: %v", configPath, err)
	}
	if len(cfg.Decisions) != len(want) {
		t.Fatalf("decision count in %s = %d, want %d", configPath, len(cfg.Decisions), len(want))
	}
	for i, wantName := range want {
		if got := cfg.Decisions[i].Name; got != wantName {
			t.Fatalf("decision[%d] in %s = %q, want %q", i, configPath, got, wantName)
		}
	}
}

func requireNamedDecision(t *testing.T, value any, want string) {
	t.Helper()

	decision, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("expected decision object, got %#v", value)
	}
	if got := decision["name"]; got != want {
		t.Fatalf("decision name = %v, want %q", got, want)
	}
}

func setupRuntimeOverrideConfigFiles(t *testing.T, sourceDecision string, runtimeDecision string) (string, string, string) {
	t.Helper()

	tempDir := t.TempDir()
	sourcePath := filepath.Join(tempDir, "config.yaml")
	runtimePath := filepath.Join(tempDir, ".vllm-sr", "runtime-config.yaml")
	if err := os.MkdirAll(filepath.Dir(runtimePath), 0o755); err != nil {
		t.Fatalf("mkdir runtime dir: %v", err)
	}

	sourceYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig(sourceDecision))
	runtimeYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig(runtimeDecision))
	if err := os.WriteFile(sourcePath, sourceYAML, 0o644); err != nil {
		t.Fatalf("write source config: %v", err)
	}
	if err := os.WriteFile(runtimePath, runtimeYAML, 0o644); err != nil {
		t.Fatalf("write runtime config: %v", err)
	}

	t.Setenv(sourceConfigPathEnv, sourcePath)
	t.Setenv(runtimeConfigPathEnv, runtimePath)
	installRuntimeSyncMirrorRunner(t, runtimePath)
	return tempDir, sourcePath, runtimePath
}

func installRuntimeSyncMirrorRunner(t *testing.T, runtimePath string) {
	t.Helper()

	previousRunner := runtimeConfigSyncRunner
	runtimeConfigSyncRunner = func(configPath string) (string, error) {
		data, err := os.ReadFile(configPath)
		if err != nil {
			return "", err
		}
		if err := os.WriteFile(runtimePath, data, 0o644); err != nil {
			return "", err
		}
		return runtimePath, nil
	}
	t.Cleanup(func() {
		runtimeConfigSyncRunner = previousRunner
	})
}

func mustMarshalDecisionTreeDeployYAML(t *testing.T, dslText string) []byte {
	t.Helper()

	compiledCfg, errs := routerdsl.Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("Compile decision-tree DSL errors: %v", errs)
	}

	canonical := config.CanonicalConfigFromRouterConfig(minimalDeployTestConfig("placeholder_route"))
	canonical.Routing = config.CanonicalRoutingFromRouterConfig(compiledCfg)
	data, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatalf("yaml.Marshal decision-tree deploy config: %v", err)
	}
	return data
}
