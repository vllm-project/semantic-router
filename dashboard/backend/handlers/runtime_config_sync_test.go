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
		cmd := exec.Command(resolved, "-c", "import yaml, jinja2")
		if err := cmd.Run(); err == nil {
			return resolved
		}
	}

	t.Fatal("python interpreter with yaml support not found")
	return ""
}

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

func TestSyncRuntimeConfigLocallySeedsKnowledgeBaseAssetsIntoRuntimeStore(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	kbDir := filepath.Join(tempDir, "knowledge_bases", "privacy")
	if err := os.MkdirAll(kbDir, 0o755); err != nil {
		t.Fatalf("mkdir kb dir: %v", err)
	}
	if err := os.WriteFile(
		filepath.Join(kbDir, "labels.json"),
		[]byte(`{"labels":{"safe":{"exemplars":["hello"]}}}`),
		0o644,
	); err != nil {
		t.Fatalf("write labels manifest: %v", err)
	}

	configYAML := `version: v0.3
global:
  model_catalog:
    kbs:
      - name: privacy_kb
        source:
          path: knowledge_bases/privacy/
          manifest: labels.json
`
	if err := os.WriteFile(configPath, []byte(configYAML), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", "/app/.vllm-sr/runtime-config.yaml")
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

	runtimeData, err := os.ReadFile(runtimePath)
	if err != nil {
		t.Fatalf("read runtime config: %v", err)
	}
	if !contains(string(runtimeData), "path: knowledge_bases/privacy/") {
		t.Fatalf("expected runtime KB path to stay aligned with source dir, got:\n%s", string(runtimeData))
	}

	stagedManifest := filepath.Join(tempDir, ".vllm-sr", "knowledge_bases", "privacy", "labels.json")
	if _, err := os.Stat(stagedManifest); err != nil {
		t.Fatalf("expected seeded knowledge base manifest, got %v", err)
	}
}

func TestSyncRuntimeConfigInManagedContainerUsesDashboardVenvPythonForSplitRuntime(t *testing.T) {
	tempDir := t.TempDir()
	dockerArgsPath := filepath.Join(tempDir, "docker-args.txt")
	dockerPath := filepath.Join(tempDir, "docker")
	dockerScript := "#!/bin/sh\n" +
		"printf '%s\n' \"$@\" > \"" + dockerArgsPath + "\"\n" +
		"printf '/app/.vllm-sr/runtime-config.yaml\n'\n"
	if err := os.WriteFile(dockerPath, []byte(dockerScript), 0o755); err != nil {
		t.Fatalf("write fake docker binary: %v", err)
	}

	t.Setenv("PATH", tempDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv(routerContainerNameEnv, "lane-a-vllm-sr-router-container")
	t.Setenv(envoyContainerNameEnv, "lane-a-vllm-sr-envoy-container")
	t.Setenv(dashboardContainerNameEnv, "lane-a-vllm-sr-dashboard-container")

	runtimePath, err := syncRuntimeConfigInManagedContainer()
	if err != nil {
		t.Fatalf("syncRuntimeConfigInManagedContainer returned error: %v", err)
	}
	if runtimePath != "/app/.vllm-sr/runtime-config.yaml" {
		t.Fatalf("runtime config path = %q", runtimePath)
	}

	argsData, err := os.ReadFile(dockerArgsPath)
	if err != nil {
		t.Fatalf("read fake docker args: %v", err)
	}
	args := strings.Split(strings.TrimSpace(string(argsData)), "\n")
	if len(args) < 4 {
		t.Fatalf("docker exec args too short: %#v", args)
	}
	if args[0] != "exec" {
		t.Fatalf("docker exec argv[0] = %q", args[0])
	}
	if args[1] != "lane-a-vllm-sr-dashboard-container" {
		t.Fatalf("managed container name = %q", args[1])
	}
	if args[2] != dashboardVenvPythonPath {
		t.Fatalf("managed split python binary = %q, want %q", args[2], dashboardVenvPythonPath)
	}
	if args[3] != "-c" {
		t.Fatalf("python exec flag = %q", args[3])
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
