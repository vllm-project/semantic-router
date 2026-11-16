package handlers

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func testRuntimeSyncPythonBinary(t *testing.T) string {
	t.Helper()

	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}

	candidates := []string{
		filepath.Join(repoRoot, ".venv-agent", "bin", "python3"),
		"python3",
		"python",
	}
	for _, candidate := range candidates {
		resolved, err := exec.LookPath(candidate)
		if err != nil {
			continue
		}
		cmd := exec.Command(resolved, "-c", "import yaml, jinja2, pydantic")
		if err := cmd.Run(); err == nil {
			return resolved
		}
	}

	// These tests drive the vllm-sr CLI through a Python interpreter. Skip rather
	// than fail where the CLI's dependencies are not installed, so `go test ./...`
	// stays usable without them; CI installs `pip install -e src/vllm-sr`, so the
	// tests do run in the Dashboard gate.
	t.Skip("python interpreter with vllm-sr CLI dependencies (yaml, jinja2, pydantic) not found")
	return ""
}

func TestConfiguredRuntimeConfigPathUsesEnvOverride(t *testing.T) {
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", "/app/.vllm-sr/runtime-config.yaml")

	got := configuredRuntimeConfigPath("/app/config.yaml")

	if got != "/app/.vllm-sr/runtime-config.yaml" {
		t.Fatalf("expected runtime config path override, got %q", got)
	}
}

func TestRuntimeOwnedConfigPathDoesNotResyncIntoNestedStateDirectory(t *testing.T) {
	stateDir := filepath.Join(t.TempDir(), ".vllm-sr")
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(stateDir, "runtime-config.stack.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.5\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", configPath)

	got, err := syncRuntimeConfigForCurrentRuntime(configPath)
	if err != nil || got != configPath {
		t.Fatalf("syncRuntimeConfigForCurrentRuntime() = %q, %v", got, err)
	}
	if _, err := os.Stat(filepath.Join(stateDir, ".vllm-sr")); !os.IsNotExist(err) {
		t.Fatalf("unexpected nested runtime state directory: %v", err)
	}
	if !isManagedContainerConfigPath(configPath) {
		t.Fatal("configured runtime path was not recognized as managed")
	}
}

func TestSyncRuntimeConfigLocallyWritesInternalRuntimeConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	configYAML := `version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
recipes:
  - name: default
    decisions:
      - name: default
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", "/app/.vllm-sr/runtime-config.yaml")
	t.Setenv("VLLM_SR_ALGORITHM_OVERRIDE", "multi_factor")
	t.Setenv("VLLM_SR_PYTHON_BIN", testRuntimeSyncPythonBinary(t))
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
	if !contains(string(runtimeData), "type: multi_factor") {
		t.Fatalf("expected the algorithm override, got:\n%s", string(runtimeData))
	}

	sourceData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read source config: %v", err)
	}
	if string(sourceData) != configYAML {
		t.Fatalf("expected source config to remain unchanged, got:\n%s", string(sourceData))
	}
}

func TestPlatformBrandingAloneDoesNotSyncRuntimeConfig(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.5\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", "")
	t.Setenv("VLLM_SR_ALGORITHM_OVERRIDE", "")
	t.Setenv("VLLM_SR_PLATFORM", "rocm")
	t.Setenv("DASHBOARD_PLATFORM", "rocm")
	// A sync would fail here: this is not a Python interpreter.
	t.Setenv("VLLM_SR_PYTHON_BIN", "/bin/sh")

	got, err := syncRuntimeConfigLocally(configPath)
	if err != nil || got != configPath {
		t.Fatalf("syncRuntimeConfigLocally() = %q, %v", got, err)
	}
}

func TestHostDashboardSavesWithoutReachingIntoContainers(t *testing.T) {
	invocations := trapContainerCLIs(t)
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.5\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", "")

	if got, err := syncRuntimeConfigForCurrentRuntime(configPath); err != nil || got != configPath {
		t.Fatalf("syncRuntimeConfigForCurrentRuntime() = %q, %v", got, err)
	}
	if err := propagateConfigToRuntime(configPath, filepath.Dir(configPath)); err != nil {
		t.Fatalf("propagateConfigToRuntime() = %v", err)
	}
	if calls := invocations(); len(calls) != 0 {
		t.Fatalf("the Dashboard ran a container CLI: %q", calls)
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
