package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercomposition"
)

func extProcAuthoringParser(t *testing.T) *config.Parser {
	t.Helper()
	compiler, err := providercomposition.NewAuthoringCompiler(
		providercatalog.BuiltinIntegrations(),
		[]providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
	)
	if err != nil {
		t.Fatalf("compose test Provider Integrations: %v", err)
	}
	return config.NewParser(compiler)
}

func parseExtProcAuthoringConfig(t *testing.T, document string) (*config.RouterConfig, error) {
	t.Helper()
	return extProcAuthoringParser(t).ParseYAMLBytes([]byte(document))
}
