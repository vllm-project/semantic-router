package routingruntime

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/backendresolver"
)

func composeProviderRegistry(
	integrations []providercatalog.Integration,
	compilers []providercatalog.BackendCompiler,
	protocolCodecs *protocolcodec.Registry,
	credentialAdapters backendresolver.StaticRegistry,
	discoveryAdapters *providerdiscovery.Registry,
) (*providercatalog.Registry, error) {
	if protocolCodecs == nil || discoveryAdapters == nil {
		return nil, fmt.Errorf("provider adapter registries are required")
	}
	wireFormats := make([]string, 0, len(protocolCodecs.Capabilities()))
	for _, capability := range protocolCodecs.Capabilities() {
		wireFormats = append(wireFormats, string(capability.Format))
	}
	discoveryValidators := discoveryAdapters.Validators()
	discoveryIDs := make([]string, 0, len(discoveryValidators))
	for _, validator := range discoveryValidators {
		discoveryIDs = append(discoveryIDs, validator.AdapterID())
	}
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations:         append([]providercatalog.Integration(nil), integrations...),
		BackendCompilers:     append([]providercatalog.BackendCompiler(nil), compilers...),
		WireFormats:          wireFormats,
		CredentialAdapterIDs: credentialAdapters.AdapterIDs(),
		DiscoveryAdapterIDs:  discoveryIDs,
	})
	if err != nil {
		return nil, fmt.Errorf("compose provider integration registry: %w", err)
	}
	return registry, nil
}
