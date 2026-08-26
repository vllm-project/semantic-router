package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

func TestResolveProductionManagementFactoryUsesRouterNativeComposition(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.AccessStore = &config.AccessStoreConfig{Type: config.AccessStoreTypePostgres}
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

func TestResolveManagementIssuerEgressPolicyLoadsSystemPolicy(t *testing.T) {
	path := filepath.Join(t.TempDir(), "issuer-egress.yaml")
	if err := os.WriteFile(path, []byte(`version: v1
schemes: [https]
hosts:
  - {host: dashboard.internal, ports: [8743], allow_cidrs: [172.31.0.0/16]}
`), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(managementIssuerEgressPolicyFileEnv, path)
	policy, err := resolveManagementIssuerEgressPolicy()
	if err != nil || policy == nil {
		t.Fatalf("resolveManagementIssuerEgressPolicy() = %v, %v", policy, err)
	}
	if _, err := policy.AuthorizeOrigin("https://dashboard.internal:8743"); err != nil {
		t.Fatalf("system issuer origin = %v", err)
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
