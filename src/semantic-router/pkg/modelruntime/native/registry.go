package native

import (
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
func (r *AdapterRegistry) Register(adapter BackendAdapter) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.adapters[adapter.Name()] = adapter
}

// Get retrieves a backend adapter by name.
func (r *AdapterRegistry) Get(name Backend) (BackendAdapter, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	adapter, ok := r.adapters[name]
	return adapter, ok
}

// List returns all registered adapters.
func (r *AdapterRegistry) List() []BackendAdapter {
	r.mu.RLock()
	defer r.mu.RUnlock()
	list := make([]BackendAdapter, 0, len(r.adapters))
	for _, adapter := range r.adapters {
		list = append(list, adapter)
	}
	return list
}

// Registry is the global adapter registry.
var Registry = NewRegistry()

