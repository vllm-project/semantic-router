package dsl

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercomposition"
)

func maintainedConfigParser(t *testing.T) *config.Parser {
	t.Helper()
	compiler, err := providercomposition.NewAuthoringCompiler(
		providercatalog.BuiltinIntegrations(),
		[]providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
	)
	if err != nil {
		t.Fatalf("compose maintained Provider Integrations: %v", err)
	}
	return config.NewParser(compiler)
}

func parseMaintainedConfig(t *testing.T, path string) *config.RouterConfig {
	t.Helper()
	cfg, err := maintainedConfigParser(t).Parse(path)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return cfg
}

func parseMaintainedConfigBytes(t *testing.T, source string, data []byte) *config.RouterConfig {
	t.Helper()
	cfg, err := maintainedConfigParser(t).ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse %s: %v", source, err)
	}
	return cfg
}
