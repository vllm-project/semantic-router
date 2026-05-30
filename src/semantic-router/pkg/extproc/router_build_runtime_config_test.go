package extproc

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestResolveInitialRouterConfigWithEmptyRuntimeRegistryParsesFileBeforeGlobal(t *testing.T) {
	globalCfg := &config.RouterConfig{
		ConfigSource: config.ConfigSourceKubernetes,
		Looper:       config.LooperConfig{GRPCMaxMsgSizeMB: 64},
	}
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(globalCfg)
	defer restoreGlobalConfig()

	configPath := filepath.Join(t.TempDir(), "router.yaml")
	writeRuntimeRegistryFileConfig(t, configPath)

	cfg, publishGlobal, err := resolveInitialRouterConfig(
		configPath,
		routerruntime.NewRegistry(nil),
	)
	if err != nil {
		t.Fatalf("resolveInitialRouterConfig() error = %v", err)
	}
	if cfg == globalCfg {
		t.Fatal("resolveInitialRouterConfig() returned legacy global config, want file config for runtime registry path")
	}
	if cfg.ConfigSource != config.ConfigSourceFile {
		t.Fatalf("resolveInitialRouterConfig() source = %q, want file source", cfg.ConfigSource)
	}
	if cfg.BackendModels.DefaultModel != "file-model" {
		t.Fatalf("resolveInitialRouterConfig() default model = %q, want file-model", cfg.BackendModels.DefaultModel)
	}
	if publishGlobal {
		t.Fatal("resolveInitialRouterConfig() publishGlobal = true, want false for runtime registry path")
	}
	if got := config.Get(); got != globalCfg {
		t.Fatalf("config.Get() = %p, want unchanged global cfg %p", got, globalCfg)
	}
}

func writeRuntimeRegistryFileConfig(t *testing.T, path string) {
	t.Helper()
	content := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: file-model
  models:
    - name: file-model
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: file-model
  decisions:
    - name: default-route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: file-model
          use_reasoning: false
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write runtime registry file config: %v", err)
	}
}
