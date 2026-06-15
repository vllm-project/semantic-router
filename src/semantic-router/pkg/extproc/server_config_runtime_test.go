package extproc

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestConfiguredGRPCMaxMessageSizePrefersRouterConfig(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		Looper: config.LooperConfig{GRPCMaxMsgSizeMB: 64},
	})
	defer restoreGlobalConfig()

	routerCfg := &config.RouterConfig{
		Looper: config.LooperConfig{GRPCMaxMsgSizeMB: 8},
	}
	server := &Server{
		service: NewRouterService(&OpenAIRouter{Config: routerCfg}),
	}

	if got, want := server.configuredGRPCMaxMessageSize(), 8*1024*1024; got != want {
		t.Fatalf("configuredGRPCMaxMessageSize() = %d, want router config size %d", got, want)
	}
}

func TestConfiguredGRPCMaxMessageSizeUsesRuntimeRegistryBeforeGlobal(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		Looper: config.LooperConfig{GRPCMaxMsgSizeMB: 64},
	})
	defer restoreGlobalConfig()

	runtimeCfg := &config.RouterConfig{
		Looper: config.LooperConfig{GRPCMaxMsgSizeMB: 16},
	}
	server := &Server{
		runtime: routerruntime.NewRegistry(runtimeCfg),
	}

	if got, want := server.configuredGRPCMaxMessageSize(), 16*1024*1024; got != want {
		t.Fatalf("configuredGRPCMaxMessageSize() = %d, want runtime config size %d", got, want)
	}
}

func TestConfiguredGRPCMaxMessageSizeWithEmptyRuntimeRegistryDoesNotUseGlobal(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		Looper: config.LooperConfig{GRPCMaxMsgSizeMB: 64},
	})
	defer restoreGlobalConfig()

	server := &Server{
		runtime: routerruntime.NewRegistry(nil),
	}

	if got, want := server.configuredGRPCMaxMessageSize(), 4*1024*1024; got != want {
		t.Fatalf("configuredGRPCMaxMessageSize() = %d, want default size %d", got, want)
	}
}

func TestConfiguredGRPCMaxMessageSizePreservesLegacyGlobalFallback(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		Looper: config.LooperConfig{GRPCMaxMsgSizeMB: 32},
	})
	defer restoreGlobalConfig()

	server := &Server{}

	if got, want := server.configuredGRPCMaxMessageSize(), 32*1024*1024; got != want {
		t.Fatalf("configuredGRPCMaxMessageSize() = %d, want global config size %d", got, want)
	}
}

func TestUsesKubernetesConfigSourcePrefersRouterConfig(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		ConfigSource: config.ConfigSourceKubernetes,
	})
	defer restoreGlobalConfig()

	server := &Server{
		service: NewRouterService(&OpenAIRouter{
			Config: &config.RouterConfig{ConfigSource: config.ConfigSourceFile},
		}),
		runtime: routerruntime.NewRegistry(&config.RouterConfig{
			ConfigSource: config.ConfigSourceKubernetes,
		}),
	}

	if server.usesKubernetesConfigSource() {
		t.Fatal("usesKubernetesConfigSource() = true, want router config to override runtime/global source")
	}
}

func TestUsesKubernetesConfigSourceUsesRuntimeRegistryBeforeGlobal(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		ConfigSource: config.ConfigSourceFile,
	})
	defer restoreGlobalConfig()

	server := &Server{
		runtime: routerruntime.NewRegistry(&config.RouterConfig{
			ConfigSource: config.ConfigSourceKubernetes,
		}),
	}

	if !server.usesKubernetesConfigSource() {
		t.Fatal("usesKubernetesConfigSource() = false, want runtime registry config source")
	}
}

func TestUsesKubernetesConfigSourceWithEmptyRuntimeRegistryDoesNotUseGlobal(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		ConfigSource: config.ConfigSourceKubernetes,
	})
	defer restoreGlobalConfig()

	server := &Server{
		runtime: routerruntime.NewRegistry(nil),
	}

	if server.usesKubernetesConfigSource() {
		t.Fatal("usesKubernetesConfigSource() = true, want empty runtime registry to avoid legacy global source")
	}
}

func TestUsesKubernetesConfigSourcePreservesLegacyGlobalFallback(t *testing.T) {
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(&config.RouterConfig{
		ConfigSource: config.ConfigSourceKubernetes,
	})
	defer restoreGlobalConfig()

	server := &Server{}

	if !server.usesKubernetesConfigSource() {
		t.Fatal("usesKubernetesConfigSource() = false, want legacy global config source")
	}
}

func replaceExtProcGlobalConfigForTest(newCfg *config.RouterConfig) func() {
	previous := config.Get()
	config.Replace(newCfg)
	return func() {
		if previous != nil {
			config.Replace(previous)
			return
		}
		config.Replace(&config.RouterConfig{})
	}
}

func TestResolveInitialRouterConfigPrefersRuntimeRegistryBeforeGlobal(t *testing.T) {
	globalCfg := &config.RouterConfig{
		ConfigSource: config.ConfigSourceKubernetes,
		Looper:       config.LooperConfig{GRPCMaxMsgSizeMB: 64},
	}
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(globalCfg)
	defer restoreGlobalConfig()

	runtimeCfg := &config.RouterConfig{
		ConfigSource: config.ConfigSourceFile,
		Looper:       config.LooperConfig{GRPCMaxMsgSizeMB: 8},
	}
	cfg, publishGlobal, err := resolveInitialRouterConfig(
		filepath.Join(t.TempDir(), "missing-config.yaml"),
		routerruntime.NewRegistry(runtimeCfg),
	)
	if err != nil {
		t.Fatalf("resolveInitialRouterConfig() error = %v", err)
	}
	if cfg != runtimeCfg {
		t.Fatalf("resolveInitialRouterConfig() cfg = %p, want runtime cfg %p", cfg, runtimeCfg)
	}
	if publishGlobal {
		t.Fatal("resolveInitialRouterConfig() publishGlobal = true, want false for runtime registry config")
	}
	if got := config.Get(); got != globalCfg {
		t.Fatalf("config.Get() = %p, want unchanged global cfg %p", got, globalCfg)
	}
}

func TestResolveInitialRouterConfigPreservesLegacyKubernetesGlobalFallback(t *testing.T) {
	globalCfg := &config.RouterConfig{
		ConfigSource: config.ConfigSourceKubernetes,
	}
	restoreGlobalConfig := replaceExtProcGlobalConfigForTest(globalCfg)
	defer restoreGlobalConfig()

	cfg, publishGlobal, err := resolveInitialRouterConfig(
		filepath.Join(t.TempDir(), "missing-config.yaml"),
		nil,
	)
	if err != nil {
		t.Fatalf("resolveInitialRouterConfig() error = %v", err)
	}
	if cfg != globalCfg {
		t.Fatalf("resolveInitialRouterConfig() cfg = %p, want global cfg %p", cfg, globalCfg)
	}
	if !publishGlobal {
		t.Fatal("resolveInitialRouterConfig() publishGlobal = false, want true for legacy config path")
	}
}
