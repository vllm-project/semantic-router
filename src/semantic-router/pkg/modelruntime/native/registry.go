package native

import (
	"fmt"
	"sort"
	"sync"
)

// AdapterRegistry manages the active backend adapters.
type AdapterRegistry struct {
	mu       sync.RWMutex
	adapters map[Backend]BackendAdapter
}

// NewRegistry creates a new registry instance.
func NewRegistry() *AdapterRegistry {
	return &AdapterRegistry{
		adapters: make(map[Backend]BackendAdapter),
	}
}

// Register adds a backend adapter to the registry.
func (r *AdapterRegistry) Register(adapter BackendAdapter) error {
	if adapter == nil {
		return fmt.Errorf("adapter cannot be nil")
	}
	name := adapter.Name()
	if name == "" {
		return fmt.Errorf("adapter name cannot be empty")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.adapters[name]; exists {
		return fmt.Errorf("backend adapter %q already registered", name)
	}

	r.adapters[name] = adapter
	return nil
}

// Get retrieves a backend adapter by name.
func (r *AdapterRegistry) Get(name Backend) (BackendAdapter, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	adapter, ok := r.adapters[name]
	return adapter, ok
}

// List returns all registered adapters in a deterministic order.
func (r *AdapterRegistry) List() []BackendAdapter {
	r.mu.RLock()
	defer r.mu.RUnlock()

	keys := make([]string, 0, len(r.adapters))
	for k := range r.adapters {
		keys = append(keys, string(k))
	}
	sort.Strings(keys)

	list := make([]BackendAdapter, 0, len(keys))
	for _, k := range keys {
		list = append(list, r.adapters[Backend(k)])
	}
	return list
}

// Registry is the global adapter registry.
var Registry = NewRegistry()

