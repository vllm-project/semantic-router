package authoring

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadRuntimeCompatibleConfigMatchesSharedFirstSliceRuntimeFixture(t *testing.T) {
	t.Parallel()

	runtimeCfg, err := LoadRuntimeCompatibleConfig(sharedFixturePath(t, "td001-first-slice-authoring.yaml"))
	if err != nil {
		t.Fatalf("LoadRuntimeCompatibleConfig() error = %v", err)
	}

	assertYAMLEqual(
		t,
		loadYAMLFile(t, sharedFixturePath(t, "td001-first-slice-runtime.yaml")),
		extractCompiledFirstSliceRuntime(runtimeCfg),
	)
}

func TestLoadRuntimeCompatibleConfigAcceptsLegacyRuntimeFixture(t *testing.T) {
	t.Parallel()

	runtimeCfg, err := LoadRuntimeCompatibleConfig(sharedFixturePath(t, "td001-first-slice-runtime.yaml"))
	if err != nil {
		t.Fatalf("LoadRuntimeCompatibleConfig() error = %v", err)
	}

	assertYAMLEqual(
		t,
		loadYAMLFile(t, sharedFixturePath(t, "td001-first-slice-runtime.yaml")),
		extractCompiledFirstSliceRuntime(runtimeCfg),
	)
}

func TestLoadRuntimeCompatibleConfigRejectsUnsupportedAuthoringTopLevelKeys(t *testing.T) {
	t.Parallel()

	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	configData := strings.TrimSpace(`
version: v0.1
listeners:
  - name: grpc-50051
    address: 0.0.0.0
    port: 50051
providers:
  models:
    - name: qwen3-4b
      endpoints:
        - name: primary
          endpoint: router.internal:8000
  default_model: qwen3-4b
memory:
  enabled: true
`) + "\n"

	if err := os.WriteFile(configPath, []byte(configData), 0o644); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}

	_, err := LoadRuntimeCompatibleConfig(configPath)
	if err == nil {
		t.Fatal("LoadRuntimeCompatibleConfig() error = nil, want unsupported authoring keys error")
	}
	if !strings.Contains(err.Error(), "unsupported top-level keys") {
		t.Fatalf("LoadRuntimeCompatibleConfig() error = %v, want unsupported key guidance", err)
	}
}
