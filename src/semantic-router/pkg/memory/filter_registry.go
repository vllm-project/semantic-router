package memory

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var (
	filterMu       sync.RWMutex
	filterRegistry = map[string]MemoryFilterFactory{}
)

func init() {
	RegisterFilter("heuristic", newHeuristicFilter)
	RegisterFilter("noop", newNoopFilter)
}

// RegisterFilter adds a named MemoryFilterFactory to the global registry.
// Calling with an existing name replaces the previous factory.
// Safe for concurrent use (intended for init-time registration).
func RegisterFilter(name string, factory MemoryFilterFactory) {
	filterMu.Lock()
	defer filterMu.Unlock()
	filterRegistry[name] = factory
}

// RegisteredFilters returns the names of all registered filter algorithms.
func RegisteredFilters() []string {
	filterMu.RLock()
	defer filterMu.RUnlock()
	names := make([]string, 0, len(filterRegistry))
	for name := range filterRegistry {
		names = append(names, name)
	}
	return names
}

// NewMemoryFilter constructs the appropriate MemoryFilter based on configuration.
// The Algorithm field in config selects the filter; defaults to "heuristic".
// If the selected algorithm is unknown, falls back to "heuristic" with a warning.
func NewMemoryFilter(global config.MemoryReflectionConfig, perDecision *config.MemoryReflectionConfig) MemoryFilter {
	cfg := resolveReflectionConfig(global, perDecision)

	if !cfg.ReflectionEnabled() {
		return &NoopFilter{}
	}

	algorithm := cfg.Algorithm
	if algorithm == "" {
		algorithm = "heuristic"
	}

	filterMu.RLock()
	factory, ok := filterRegistry[algorithm]
	filterMu.RUnlock()

	if !ok {
		logging.Warnf("MemoryFilter: unknown algorithm %q, falling back to heuristic", algorithm)
		filterMu.RLock()
		factory = filterRegistry["heuristic"]
		filterMu.RUnlock()
	}

	f := factory(global, perDecision)
	if f == nil {
		return &NoopFilter{}
	}
	return f
}

// ---------------------------------------------------------------------------
// NoopFilter -- passes all memories through unmodified.
// ---------------------------------------------------------------------------

type NoopFilter struct{}

func (n *NoopFilter) Filter(memories []*RetrieveResult) []*RetrieveResult {
	return memories
}

func newNoopFilter(_ config.MemoryReflectionConfig, _ *config.MemoryReflectionConfig) MemoryFilter {
	return &NoopFilter{}
}

// ---------------------------------------------------------------------------
// ChainFilter -- composes multiple MemoryFilters sequentially.
// Each filter's output is the next filter's input.
// ---------------------------------------------------------------------------

type ChainFilter struct {
	filters []MemoryFilter
}

// NewChainFilter creates a filter that runs each inner filter in order.
// Nil entries are silently skipped.
func NewChainFilter(filters ...MemoryFilter) *ChainFilter {
	nonNil := make([]MemoryFilter, 0, len(filters))
	for _, f := range filters {
		if f != nil {
			nonNil = append(nonNil, f)
		}
	}
	return &ChainFilter{filters: nonNil}
}

func (c *ChainFilter) Filter(memories []*RetrieveResult) []*RetrieveResult {
	for _, f := range c.filters {
		memories = f.Filter(memories)
		if len(memories) == 0 {
			return memories
		}
	}
	return memories
}
