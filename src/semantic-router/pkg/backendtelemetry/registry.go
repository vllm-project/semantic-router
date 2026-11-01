package backendtelemetry

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Target is one replica an adapter collects from.
type Target struct {
	Identity Identity
	// MetricsURL is the absolute URL the adapter reads, such as
	// http://10.0.0.1:8000/metrics.
	MetricsURL string
}

// AdapterConfig configures one adapter instance.
type AdapterConfig struct {
	Targets []Target
	// RequestTimeout bounds one scrape of one target.
	RequestTimeout time.Duration
	// TTL is stamped on the adapter's observations; zero leaves the store's
	// default in effect.
	TTL time.Duration
}

// Adapter collects observations from one engine's replicas. Collect runs off
// the request path. It returns every observation it could build even when
// some targets fail, together with the joined errors of the failures.
type Adapter interface {
	Engine() Engine
	Collect(ctx context.Context) ([]Observation, error)
}

// Constructor builds an adapter from its configuration.
type Constructor func(AdapterConfig) (Adapter, error)

// Registry maps each engine to the constructor of its adapter. It is safe for
// concurrent use.
type Registry struct {
	mu           sync.RWMutex
	constructors map[Engine]Constructor
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{constructors: make(map[Engine]Constructor)}
}

// Register adds the constructor for an engine. Registering an unknown engine,
// a nil constructor, or a second constructor for the same engine fails.
func (r *Registry) Register(engine Engine, constructor Constructor) error {
	if _, err := ParseEngine(string(engine)); err != nil {
		return err
	}
	if constructor == nil {
		return fmt.Errorf("engine %s: constructor is nil", engine)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.constructors[engine]; ok {
		return fmt.Errorf("engine %s: adapter already registered", engine)
	}
	r.constructors[engine] = constructor
	return nil
}

// ErrNoAdapter reports an engine without a registered adapter.
var ErrNoAdapter = errors.New("no adapter registered")

// New builds the adapter for an engine.
func (r *Registry) New(engine Engine, cfg AdapterConfig) (Adapter, error) {
	r.mu.RLock()
	constructor, ok := r.constructors[engine]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("engine %s: %w", engine, ErrNoAdapter)
	}
	adapter, err := constructor(cfg)
	if err != nil {
		return nil, fmt.Errorf("engine %s: %w", engine, err)
	}
	if adapter.Engine() != engine {
		return nil, fmt.Errorf("engine %s: constructor built an adapter for %s", engine, adapter.Engine())
	}
	return adapter, nil
}

// Has reports whether an engine has a registered adapter.
func (r *Registry) Has(engine Engine) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.constructors[engine]
	return ok
}
