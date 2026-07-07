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
	EngineKind() EngineKind
	Collect(ctx context.Context) ([]BackendTelemetry, error)
}

// AdapterConstructor creates a telemetry adapter.
type AdapterConstructor func(AdapterConfig) (TelemetryAdapter, error)

// Registry stores telemetry adapter constructors by engine kind.
type Registry struct {
	mu           sync.RWMutex
	constructors map[EngineKind]AdapterConstructor
}

var defaultRegistry = NewRegistry()

// NewRegistry creates an empty adapter registry.
func NewRegistry() *Registry {
	return &Registry{constructors: map[EngineKind]AdapterConstructor{}}
}

// RegisterAdapter registers an adapter constructor on the package-level registry.
func RegisterAdapter(kind EngineKind, constructor AdapterConstructor) error {
	return defaultRegistry.Register(kind, constructor)
}

// NewAdapter constructs an adapter from the package-level registry.
func NewAdapter(kind EngineKind, cfg AdapterConfig) (TelemetryAdapter, error) {
	return defaultRegistry.Create(kind, cfg)
}

// AdapterRegistered reports whether an engine kind has a package-level constructor.
func AdapterRegistered(kind EngineKind) bool {
	return defaultRegistry.Has(kind)
}

// Register associates an engine kind with an adapter constructor.
func (r *Registry) Register(kind EngineKind, constructor AdapterConstructor) error {
	if r == nil {
		return fmt.Errorf("backend telemetry adapter registry is nil")
	}
	if kind == "" {
		return fmt.Errorf("backend telemetry adapter engine kind is required")
	}
	if constructor == nil {
		return fmt.Errorf("backend telemetry adapter constructor is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.constructors[kind] = constructor
	return nil
}

// Create builds an adapter for an engine kind.
func (r *Registry) Create(kind EngineKind, cfg AdapterConfig) (TelemetryAdapter, error) {
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

// Has reports whether an engine kind is registered.
func (r *Registry) Has(kind EngineKind) bool {
	if r == nil {
		return false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.constructors[kind]
	return ok
}
