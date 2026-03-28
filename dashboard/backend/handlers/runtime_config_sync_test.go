package handlers

import (
	"os"
	"os/exec"
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
	pythonBinary := "python3"
	if _, err := exec.LookPath(pythonBinary); err != nil {
		pythonBinary = "python"
	}
	t.Setenv("VLLM_SR_PYTHON_BIN", pythonBinary)
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

func TestRuntimeSyncPythonBinaryPrefersVirtualEnvPython(t *testing.T) {
	venvDir := t.TempDir()
	pythonDir := filepath.Join(venvDir, "bin")
	if err := os.MkdirAll(pythonDir, 0o755); err != nil {
		t.Fatalf("create venv bin dir: %v", err)
	}

	pythonPath := filepath.Join(pythonDir, "python3")
	script := "#!/bin/sh\nexit 0\n"
	if err := os.WriteFile(pythonPath, []byte(script), 0o755); err != nil {
		t.Fatalf("write venv python shim: %v", err)
	}

	t.Setenv("VIRTUAL_ENV", venvDir)
	t.Setenv("VLLM_SR_PYTHON_BIN", "")

	resolved, err := runtimeSyncPythonBinary()
	if err != nil {
		t.Fatalf("runtimeSyncPythonBinary returned error: %v", err)
	}
	if resolved != pythonPath {
		t.Fatalf("expected virtualenv python %q, got %q", pythonPath, resolved)
	}
}
