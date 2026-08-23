// Package providercomposition wires Provider Integration metadata to the
// codec and adapter capabilities installed by an application. The catalog remains
// extensible while callers share one fail-closed authoring compiler setup.
package providercomposition

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential/backendresolver"
)

// NewAuthoringCompiler composes the caller's Provider Integrations and backend
// compilers with the wire codecs, credential, and discovery adapters installed in
// this binary. Product metadata remains an application concern and never
// enters the Router data plane.
func NewAuthoringCompiler(
	integrations []providercatalog.Integration,
	compilers []providercatalog.BackendCompiler,
) (modelauthoring.ConnectionCompiler, error) {
	codecRegistry := protocolcodec.NewBuiltinRegistry()
	wireFormats := make([]string, 0, len(codecRegistry.Capabilities()))
	for _, capability := range codecRegistry.Capabilities() {
		wireFormats = append(wireFormats, string(capability.Format))
	}

	credentialRegistry, err := backendresolver.BuiltinRegistry()
	if err != nil {
		return nil, fmt.Errorf("compose Provider credential adapters: %w", err)
	}
	discoveryRegistry, err := providerdiscovery.BuiltinRegistry()
	if err != nil {
		return nil, fmt.Errorf("compose Provider discovery adapters: %w", err)
	}
	discoveryIDs := make([]string, 0, len(discoveryRegistry.Validators()))
	for _, validator := range discoveryRegistry.Validators() {
		discoveryIDs = append(discoveryIDs, validator.AdapterID())
	}

	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations:         integrations,
		BackendCompilers:     compilers,
		WireFormats:          wireFormats,
		CredentialAdapterIDs: credentialRegistry.AdapterIDs(),
		DiscoveryAdapterIDs:  discoveryIDs,
	})
	if err != nil {
		return nil, fmt.Errorf("compose Provider Integration registry: %w", err)
	}
	return providercatalog.AuthoringCompiler{Registry: registry}, nil
}
