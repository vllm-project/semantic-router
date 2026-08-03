package backend

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// AdapterTarget describes one metrics collection target for an adapter.
type AdapterTarget struct {
	Identity        BackendIdentity
	MetricsEndpoint string
	Headers         map[string]string
	Labels          map[string]string
}

// AdapterConfig carries engine-neutral adapter construction settings.
type AdapterConfig struct {
	Targets  []AdapterTarget
	Store    *Store
	Interval time.Duration
	Labels   map[string]string
}

// TelemetryAdapter normalizes engine-specific metrics into BackendTelemetry.
type TelemetryAdapter interface {
	Runtime() Runtime
	Collect(ctx context.Context) ([]BackendTelemetry, error)
}

// AdapterConstructor creates a telemetry adapter.
type AdapterConstructor func(AdapterConfig) (TelemetryAdapter, error)

// Registry stores telemetry adapter constructors by runtime.
type Registry struct {
	mu           sync.RWMutex
	constructors map[Runtime]AdapterConstructor
}

var defaultRegistry = NewRegistry()

// NewRegistry creates an empty adapter registry.
func NewRegistry() *Registry {
	return &Registry{constructors: map[Runtime]AdapterConstructor{}}
}

// RegisterAdapter registers an adapter constructor on the package-level registry.
func RegisterAdapter(kind Runtime, constructor AdapterConstructor) error {
	return defaultRegistry.Register(kind, constructor)
}

// NewAdapter constructs an adapter from the package-level registry.
func NewAdapter(kind Runtime, cfg AdapterConfig) (TelemetryAdapter, error) {
	return defaultRegistry.Create(kind, cfg)
}

// AdapterRegistered reports whether a runtime has a package-level constructor.
func AdapterRegistered(kind Runtime) bool {
	return defaultRegistry.Has(kind)
}

// Register associates a runtime with an adapter constructor.
func (r *Registry) Register(kind Runtime, constructor AdapterConstructor) error {
	if r == nil {
		return fmt.Errorf("backend telemetry adapter registry is nil")
	}
	if kind == "" {
		return fmt.Errorf("backend telemetry adapter runtime is required")
	}
	if constructor == nil {
		return fmt.Errorf("backend telemetry adapter constructor is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.constructors[kind] = constructor
	return nil
}

// Create builds an adapter for a runtime.
func (r *Registry) Create(kind Runtime, cfg AdapterConfig) (TelemetryAdapter, error) {
	if r == nil {
		return nil, fmt.Errorf("backend telemetry adapter registry is nil")
	}

	r.mu.RLock()
	constructor, ok := r.constructors[kind]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("backend telemetry adapter %q is not registered", kind)
	}
	return constructor(cfg)
}

// Has reports whether a runtime is registered.
func (r *Registry) Has(kind Runtime) bool {
	if r == nil {
		return false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.constructors[kind]
	return ok
}
