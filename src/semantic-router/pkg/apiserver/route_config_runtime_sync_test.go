//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
		if err := os.WriteFile(filepath.Clean(runtimePath), data, 0o644); err != nil { //nolint:gosec // G703: path from t.TempDir()
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
		if err := os.WriteFile(filepath.Clean(runtimePath), data, 0o644); err != nil { //nolint:gosec // G703: path from t.TempDir()
			return "", err
		}
		return runtimePath, nil
	}
	t.Cleanup(func() {
		runtimeConfigSyncRunner = previousRunner
	})
}
