package main

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercomposition"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
)

// productionProviderIntegrations is the Router application's Provider
// extension boundary. Durable routing runtime receives typed integrations and
// compilers; it does not select product metadata itself.
func productionProviderIntegrations() (
	[]providercatalog.Integration,
	[]providercatalog.BackendCompiler,
) {
	return providercatalog.BuiltinIntegrations(), []providercatalog.BackendCompiler{
		providercatalog.StaticBackendCompiler{},
	}
}

func productionModelConnectionCompiler() (modelauthoring.ConnectionCompiler, error) {
	integrations, compilers := productionProviderIntegrations()
	return providercomposition.NewAuthoringCompiler(integrations, compilers)
}
