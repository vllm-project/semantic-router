package providerdiscovery

import (
	"fmt"
	"regexp"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
)

var adapterIDPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)

// Registry is immutable after construction and can be shared across
// Management replicas. New provider products do not add entries here; only a
// genuinely new discovery wire protocol does.
type Registry struct {
	adapters map[string]Adapter
}

func NewRegistry(adapters []Adapter) (*Registry, error) {
	if len(adapters) == 0 {
		return nil, fmt.Errorf("%w: at least one discovery adapter is required", ErrAdapterUnavailable)
	}
	registry := &Registry{adapters: make(map[string]Adapter, len(adapters))}
	for index, adapter := range adapters {
		if adapter == nil || !adapterIDPattern.MatchString(adapter.AdapterID()) {
			return nil, fmt.Errorf("%w: adapter %d has an invalid identity", ErrAdapterUnavailable, index)
		}
		if _, duplicate := registry.adapters[adapter.AdapterID()]; duplicate {
			return nil, fmt.Errorf("%w: adapter %q is registered more than once", ErrAdapterUnavailable, adapter.AdapterID())
		}
		registry.adapters[adapter.AdapterID()] = adapter
	}
	return registry, nil
}

func BuiltinRegistry() (*Registry, error) {
	return NewRegistry([]Adapter{OpenAIModelsAdapter{}, AnthropicModelsAdapter{}})
}

func (registry *Registry) Adapter(adapterID string) (Adapter, error) {
	if registry == nil {
		return nil, ErrAdapterUnavailable
	}
	adapter, found := registry.adapters[adapterID]
	if !found {
		return nil, fmt.Errorf("%w: %q", ErrAdapterUnavailable, adapterID)
	}
	return adapter, nil
}

func (registry *Registry) Validators() []providercatalog.DiscoveryRequestValidator {
	if registry == nil {
		return nil
	}
	ids := make([]string, 0, len(registry.adapters))
	for adapterID := range registry.adapters {
		ids = append(ids, adapterID)
	}
	sort.Strings(ids)
	result := make([]providercatalog.DiscoveryRequestValidator, 0, len(ids))
	for _, adapterID := range ids {
		result = append(result, registry.adapters[adapterID])
	}
	return result
}
