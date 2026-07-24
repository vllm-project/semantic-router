package native

import (
	"context"
	"fmt"
	"sync"
)

type AdapterRegistry struct {
	adapters map[Backend]BackendAdapter
	mu       sync.RWMutex
}

func NewAdapterRegistry() *AdapterRegistry {
	return &AdapterRegistry{
		adapters: make(map[Backend]BackendAdapter),
	}
}

func (r *AdapterRegistry) Register(adapter BackendAdapter) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.adapters[adapter.Name()] = adapter
}

func (r *AdapterRegistry) Get(backend Backend) (BackendAdapter, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	adapter, ok := r.adapters[backend]
	return adapter, ok
}

func (r *AdapterRegistry) List() []BackendAdapter {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var list []BackendAdapter
	for _, adapter := range r.adapters {
		list = append(list, adapter)
	}
	return list
}

func (r *AdapterRegistry) FindSupportingAdapter(cap Capability, fam Family) (BackendAdapter, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, adapter := range r.adapters {
		caps := adapter.Capabilities()
		hasCap := false
		hasFam := false
		for _, c := range caps.Capabilities {
			if c == cap {
				hasCap = true
				break
			}
		}
		for _, f := range caps.SupportedFamilies {
			if f == fam {
				hasFam = true
				break
			}
		}
		if hasCap && hasFam {
			return adapter, nil
		}
	}
	return nil, fmt.Errorf("no adapter supports capability %s and family %s", cap, fam)
}

var Registry = NewAdapterRegistry()

func LoadModel(ctx context.Context, backend Backend, req LoadRequest) (ModelHandle, error) {
	adapter, ok := Registry.Get(backend)
	if !ok {
		return nil, fmt.Errorf("backend %s not found in registry", backend)
	}
	return adapter.LoadModel(ctx, req)
}
