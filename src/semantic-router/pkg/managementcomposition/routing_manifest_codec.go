package managementcomposition

import (
	"fmt"
	"reflect"
	"strings"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type v03RoutingManifestCodec struct {
	compiler providercatalog.AuthoringCompiler
}

func newV03RoutingManifestCodec(registry *providercatalog.Registry) (*v03RoutingManifestCodec, error) {
	if registry == nil {
		return nil, fmt.Errorf("routing manifest Provider Integration registry is required")
	}
	compiler := providercatalog.AuthoringCompiler{Registry: registry}
	return &v03RoutingManifestCodec{compiler: compiler}, nil
}

func mustRoutingManifestCodec(registry *providercatalog.Registry) *v03RoutingManifestCodec {
	codec, err := newV03RoutingManifestCodec(registry)
	if err != nil {
		panic(err)
	}
	return codec
}

func (codec *v03RoutingManifestCodec) Decode(document []byte) (*routingsnapshot.Snapshot, error) {
	if codec == nil || codec.compiler.Registry == nil {
		return nil, fmt.Errorf("routing manifest codec is unavailable")
	}
	source, err := decodeRoutingManifestSource(document)
	if err != nil {
		return nil, err
	}
	snapshot, err := config.CompileFileRoutingSnapshot(source, codec.compiler)
	if err != nil {
		return nil, err
	}
	if snapshot == nil || len(snapshot.Models) == 0 || len(snapshot.Recipes) == 0 || len(snapshot.Entrypoints) == 0 {
		return nil, fmt.Errorf("v0.3 routing manifest has no complete routing closure")
	}
	return snapshot, nil
}

func decodeRoutingManifestSource(document []byte) (config.CanonicalConfig, error) {
	var source config.CanonicalConfig
	if err := config.DecodeYAML12Strict(document, &source); err != nil {
		return config.CanonicalConfig{}, fmt.Errorf("decode strict v0.3 routing manifest: %w", err)
	}
	if source.Version != "v0.3" || len(source.Listeners) != 0 || !routingOnlyGlobal(source.Global) {
		return config.CanonicalConfig{}, fmt.Errorf("routing import accepts only v0.3 routing and optional billing currency")
	}
	for _, model := range source.Providers.Models {
		for _, backend := range model.BackendRefs {
			if backend.APIKey != "" || backend.APIKeyEnv != "" {
				return config.CanonicalConfig{}, fmt.Errorf("routing import never accepts Provider secret material or environment secret sources")
			}
			if backend.Credential != strings.TrimSpace(backend.Credential) {
				return config.CanonicalConfig{}, fmt.Errorf("routing credential references must be canonical")
			}
		}
	}
	return source, nil
}

func (codec *v03RoutingManifestCodec) Encode(snapshot *routingsnapshot.Snapshot) ([]byte, error) {
	if codec == nil || codec.compiler.Registry == nil {
		return nil, fmt.Errorf("routing manifest codec is unavailable")
	}
	manifest, err := config.CanonicalRoutingManifestFromSnapshot(snapshot)
	if err != nil {
		return nil, err
	}
	catalog := codec.compiler.Registry.Snapshot()
	if catalog == nil {
		return nil, fmt.Errorf("provider Integration catalog is unavailable")
	}
	for modelIndex := range manifest.Providers.Models {
		model := &manifest.Providers.Models[modelIndex]
		for backendIndex := range model.BackendRefs {
			backend := &model.BackendRefs[backendIndex]
			provider, found := catalog.Get(backend.Provider)
			if !found {
				return nil, fmt.Errorf("provider Integration %q is unavailable", backend.Provider)
			}
			apiStyle, err := exportedProviderAPIStyle(provider, backend.Type)
			if err != nil {
				return nil, err
			}
			backend.Type = apiStyle
		}
	}
	return yaml.Marshal(manifest)
}

func exportedProviderAPIStyle(provider providercatalog.Definition, wireFormat string) (string, error) {
	matched := ""
	for _, candidate := range provider.Interfaces {
		if string(candidate.WireFormat) != wireFormat {
			continue
		}
		if matched != "" {
			return "", fmt.Errorf("provider %q has ambiguous API styles for exported wire format", provider.ID)
		}
		matched = candidate.ID
	}
	if matched == "" {
		return "", fmt.Errorf("provider %q cannot export wire format %q", provider.ID, wireFormat)
	}
	return matched, nil
}

func routingOnlyGlobal(global *config.CanonicalGlobal) bool {
	if global == nil {
		return true
	}
	return reflect.DeepEqual(global, &config.CanonicalGlobal{Billing: global.Billing})
}
