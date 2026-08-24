package main

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

func TestResolveProductionManagementFactoryUsesRouterNativeComposition(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.ControlPlane.Mode = config.ControlPlaneModeManaged
	cfg.Agent.PublicInferenceEndpoint = "http://public-inference.internal/v1/chat/completions"
	cfg.ManagementAPI.Auth.Mode = config.ManagementAuthModeRouter
	cfg.ManagementAPI.Auth.Roles = nil
	cfg.AccessRuntimeStore = &config.AccessRuntimeStoreConfig{
		Type:  config.AccessRuntimeStoreTypeRedis,
		Redis: config.RedisAccessRuntimeStoreConfig{KeyPrefix: "vllm-sr:test:"},
	}
	factory, err := resolveProductionManagementFactory(&cfg)
	if err != nil || factory == nil {
		t.Fatalf("resolveProductionManagementFactory() = %v, %v", factory, err)
	}

	cfg.ManagementAPI.Auth.Mode = config.ManagementAuthModeBearer
	if factory, err = resolveProductionManagementFactory(&cfg); err == nil || factory != nil {
		t.Fatalf("legacy bearer factory = %v, %v; want fail closed", factory, err)
	}
}

func TestProductionProviderIntegrationsAreExplicitTypedApplicationInput(t *testing.T) {
	integrations, compilers := productionProviderIntegrations()
	if len(integrations) == 0 || len(compilers) == 0 {
		t.Fatal("production Provider integration set is empty")
	}
	if compilers[0].AdapterID() != providercatalog.StaticBackendCompilerID {
		t.Fatalf("production compiler = %q", compilers[0].AdapterID())
	}
	first := integrations[0].Definition()
	if first.ID == "" || first.Revision != "" {
		t.Fatalf("application integration = %+v", first)
	}
}
