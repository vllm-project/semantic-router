package handlers

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConfiguredRuntimeConfigPathUsesEnvOverride(t *testing.T) {
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", "/app/.vllm-sr/runtime-config.yaml")

	got := configuredRuntimeConfigPath("/app/config.yaml")

	if got != "/app/.vllm-sr/runtime-config.yaml" {
		t.Fatalf("expected runtime config path override, got %q", got)
	}
}

func TestSyncRuntimeConfigLocallyWritesInternalRuntimeConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	configYAML := `version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
global:
  model_catalog:
    embeddings:
      semantic:
        use_cpu: true
        embedding_config:
          model_type: mmbert
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", "/app/.vllm-sr/runtime-config.yaml")
	t.Setenv("DASHBOARD_PLATFORM", "amd")
	t.Setenv("VLLM_SR_PYTHON_BIN", "python")
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	t.Setenv("VLLM_SR_CLI_PATH", filepath.Join(repoRoot, "src", "vllm-sr"))

	runtimePath, err := syncRuntimeConfigLocally(configPath)
	if err != nil {
		t.Fatalf("syncRuntimeConfigLocally returned error: %v", err)
	}

	expectedRuntimePath := filepath.Join(tempDir, ".vllm-sr", "runtime-config.yaml")
	if runtimePath != expectedRuntimePath {
		t.Fatalf("expected runtime path %q, got %q", expectedRuntimePath, runtimePath)
	}

	runtimeData, err := os.ReadFile(runtimePath)
	if err != nil {
		t.Fatalf("read runtime config: %v", err)
	}
	if !contains(string(runtimeData), "use_cpu: false") {
		t.Fatalf("expected AMD runtime override to force GPU defaults, got:\n%s", string(runtimeData))
	}

	sourceData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read source config: %v", err)
	}
	if !contains(string(sourceData), "use_cpu: true") {
		t.Fatalf("expected source config to remain unchanged, got:\n%s", string(sourceData))
	}
}

func TestRuntimeSyncPythonBinaryRejectsNonPythonOverride(t *testing.T) {
	t.Setenv("VLLM_SR_PYTHON_BIN", "/bin/sh")

	_, err := runtimeSyncPythonBinary()
	if err == nil {
		t.Fatal("expected non-python override to fail")
	}
	if !strings.Contains(err.Error(), "unsupported runtime sync python binary") {
		t.Fatalf("expected unsupported binary error, got %v", err)
	}
}
