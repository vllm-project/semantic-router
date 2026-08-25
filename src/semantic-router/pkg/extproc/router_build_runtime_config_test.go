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
		Looper: config.LooperConfig{GRPCMaxMsgSizeMB: 64},
	}
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(globalCfg)
	defer restoreGlobalConfig()

	configPath := filepath.Join(t.TempDir(), "router.yaml")
	writeRuntimeRegistryFileConfig(t, configPath)

	cfg, publishGlobal, err := resolveInitialRouterConfig(
		configPath,
		routerruntime.NewRegistry(nil),
		extProcAuthoringParser(t).Parse,
	)
	if err != nil {
		t.Fatalf("resolveInitialRouterConfig() error = %v", err)
	}
	if cfg == globalCfg {
		t.Fatal("resolveInitialRouterConfig() returned legacy global config, want file config for runtime registry path")
	}
	if _, ok := cfg.ModelConfig["file-model"]; !ok {
		t.Fatalf("resolveInitialRouterConfig() Models = %+v, want file-model", cfg.ModelConfig)
	}
	if _, ok := cfg.RecipeForRequestModel("router/file"); !ok {
		t.Fatal("resolveInitialRouterConfig() did not compile the file Entrypoint")
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
  models:
    - name: file-model
      provider_model_id: file-model
      backend_refs:
        - provider: vllm
          endpoint: http://127.0.0.1:8000
routing:
  modelCards:
    - name: file-model
recipes:
  - name: default
    routing:
      decisions:
        - name: default-route
          priority: 1
          rules: {operator: AND, conditions: []}
entrypoints:
  - model_names: [router/file]
    recipe: default
    assignments:
      default-route:
        models: [{model: file-model}]
`)
	if err := os.WriteFile(path, content, 0o644); err != nil {
		t.Fatalf("write runtime registry file config: %v", err)
	}
}
