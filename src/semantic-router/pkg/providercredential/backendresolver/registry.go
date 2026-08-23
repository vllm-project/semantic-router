package backendresolver

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

var registryAdapterPattern = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)

// StaticRegistry is an immutable startup-validated provider adapter registry.
// Provider catalog construction is the sole place that decides which adapter a
// provider uses.
type StaticRegistry struct {
	adapters map[string]Materializer
}

func NewStaticRegistry(adapters map[string]Materializer) (StaticRegistry, error) {
	if len(adapters) == 0 {
		return StaticRegistry{}, fmt.Errorf("at least one provider credential adapter is required")
	}
	copy := make(map[string]Materializer, len(adapters))
	for adapterID, adapter := range adapters {
		if !registryAdapterPattern.MatchString(adapterID) || adapter == nil {
			return StaticRegistry{}, fmt.Errorf("provider credential adapter registry is invalid")
		}
		copy[adapterID] = adapter
	}
	return StaticRegistry{adapters: copy}, nil
}

func (r StaticRegistry) ForAdapter(adapterID string) (Materializer, error) {
	if strings.TrimSpace(adapterID) != adapterID {
		return nil, fmt.Errorf("provider credential adapter id is not canonical")
	}
	adapter, found := r.adapters[adapterID]
	if !found {
		return nil, fmt.Errorf("provider credential adapter %q is not registered", adapterID)
	}
	return adapter, nil
}

func (r StaticRegistry) AdapterIDs() []string {
	result := make([]string, 0, len(r.adapters))
	for adapterID := range r.adapters {
		result = append(result, adapterID)
	}
	sort.Strings(result)
	return result
}

// BuiltinRegistry contains only stable wire-authentication mechanisms. Product
// providers are control-plane manifests and never appear in this runtime list.
func BuiltinRegistry() (StaticRegistry, error) {
	return NewStaticRegistry(map[string]Materializer{
		"bearer":    HeaderMaterializer{Header: "Authorization", Prefix: "Bearer "},
		"x-api-key": HeaderMaterializer{Header: "X-Api-Key"},
		"api-key":   HeaderMaterializer{Header: "Api-Key"},
	})
}
