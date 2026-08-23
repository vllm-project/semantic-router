package providercatalog

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
)

const registryCapabilityDigestSchema = "provider-catalog.registry-capabilities.v1"

type RegistryOptions struct {
	Integrations         []Integration
	BackendCompilers     []BackendCompiler
	WireFormats          []string
	CredentialAdapterIDs []string
	DiscoveryAdapterIDs  []string
}

// Registry is the immutable control-plane compilation registry. Product
// integrations never cross its routing publication boundary.
type Registry struct {
	snapshot      *Snapshot
	compiler      map[string]BackendCompiler
	wireFormats   map[string]struct{}
	credential    map[string]struct{}
	discovery     map[string]struct{}
	controlDigest [sha256.Size]byte
	dataDigest    [sha256.Size]byte
}

func NewRegistry(options RegistryOptions) (*Registry, error) {
	compilerIDs, compilers, err := canonicalCompilerSet(options.BackendCompilers)
	if err != nil {
		return nil, err
	}
	if len(compilerIDs) == 0 {
		return nil, fmt.Errorf("at least one backend compiler is required")
	}
	wireFormatIDs, wireFormats, err := canonicalAdapterSet("wire format", options.WireFormats)
	if err != nil {
		return nil, err
	}
	if len(wireFormatIDs) == 0 {
		return nil, fmt.Errorf("at least one wire format is required")
	}
	credentialIDs, credentials, err := canonicalAdapterSet("credential", options.CredentialAdapterIDs)
	if err != nil {
		return nil, err
	}
	discoveryIDs, discoveries, err := canonicalAdapterSet("discovery", options.DiscoveryAdapterIDs)
	if err != nil {
		return nil, err
	}
	registry := &Registry{compiler: compilers, wireFormats: wireFormats, credential: credentials, discovery: discoveries}
	registry.controlDigest, err = capabilityDigest(CapabilityPlaneControl, compilerIDs, discoveryIDs)
	if err != nil {
		return nil, err
	}
	registry.dataDigest, err = capabilityDigest(CapabilityPlaneData, wireFormatIDs, credentialIDs)
	if err != nil {
		return nil, err
	}
	definitions, err := evaluateIntegrations(options.Integrations)
	if err != nil {
		return nil, err
	}
	registry.snapshot, err = buildSnapshot(definitions, registry, false)
	if err != nil {
		return nil, err
	}
	return registry, nil
}

func evaluateIntegrations(integrations []Integration) ([]Definition, error) {
	if len(integrations) == 0 {
		return nil, fmt.Errorf("at least one provider integration is required")
	}
	definitions := make([]Definition, 0, len(integrations))
	for index, integration := range integrations {
		if nilIntegration(integration) {
			return nil, fmt.Errorf("provider integration %d is nil", index)
		}
		definitions = append(definitions, integration.Definition())
	}
	return definitions, nil
}

func nilIntegration(integration Integration) bool {
	return nilInterface(integration)
}

func nilInterface(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}

func (registry *Registry) Snapshot() *Snapshot {
	if registry == nil {
		return nil
	}
	return cloneSnapshot(registry.snapshot)
}

func (registry *Registry) BackendCompiler(id string) (BackendCompiler, bool) {
	if registry == nil {
		return nil, false
	}
	compiler, found := registry.compiler[id]
	return compiler, found
}

func (registry *Registry) HasWireFormat(id string) bool {
	if registry == nil {
		return false
	}
	_, ok := registry.wireFormats[id]
	return ok
}

func (registry *Registry) HasCredentialAdapter(id string) bool {
	if registry == nil {
		return false
	}
	_, ok := registry.credential[id]
	return ok
}

func (registry *Registry) HasDiscoveryAdapter(id string) bool {
	if registry == nil {
		return false
	}
	_, ok := registry.discovery[id]
	return ok
}

func (registry *Registry) CapabilityDigest(plane CapabilityPlane) ([]byte, error) {
	if registry == nil {
		return nil, fmt.Errorf("provider integration registry is unavailable")
	}
	switch plane {
	case CapabilityPlaneControl:
		return append([]byte(nil), registry.controlDigest[:]...), nil
	case CapabilityPlaneData:
		return append([]byte(nil), registry.dataDigest[:]...), nil
	default:
		return nil, fmt.Errorf("provider Catalog capability plane is invalid")
	}
}

func capabilityDigest(plane CapabilityPlane, first, second []string) ([sha256.Size]byte, error) {
	payload, err := json.Marshal(struct {
		Schema string   `json:"schema"`
		Plane  string   `json:"plane"`
		First  []string `json:"first"`
		Second []string `json:"second"`
	}{
		registryCapabilityDigestSchema, string(plane), first, second,
	})
	if err != nil {
		return [sha256.Size]byte{}, fmt.Errorf("encode %s Provider Catalog capabilities: %w", plane, err)
	}
	return sha256.Sum256(payload), nil
}

func canonicalAdapterSet(kind string, input []string) ([]string, map[string]struct{}, error) {
	ids := append([]string(nil), input...)
	sort.Strings(ids)
	set := make(map[string]struct{}, len(ids))
	for index, id := range ids {
		if !idPattern.MatchString(id) {
			return nil, nil, fmt.Errorf("%s adapter %q has an invalid identity", kind, id)
		}
		if index > 0 && ids[index-1] == id {
			return nil, nil, fmt.Errorf("%s adapter %q is duplicated", kind, id)
		}
		set[id] = struct{}{}
	}
	return ids, set, nil
}

func canonicalCompilerSet(input []BackendCompiler) ([]string, map[string]BackendCompiler, error) {
	set := make(map[string]BackendCompiler, len(input))
	ids := make([]string, 0, len(input))
	for index, compiler := range input {
		if nilInterface(compiler) || !idPattern.MatchString(compiler.AdapterID()) {
			return nil, nil, fmt.Errorf("backend compiler %d has an invalid identity", index)
		}
		id := compiler.AdapterID()
		if _, found := set[id]; found {
			return nil, nil, fmt.Errorf("backend compiler %q is duplicated", id)
		}
		set[id] = compiler
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids, set, nil
}

var _ adapterCapabilities = (*Registry)(nil)
