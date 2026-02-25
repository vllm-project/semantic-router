package memory

import (
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func sampleMemories() []*RetrieveResult {
	now := time.Now()
	return []*RetrieveResult{
		{Memory: &Memory{ID: "m1", Content: "User prefers Go", CreatedAt: now}, Score: 0.9},
		{Memory: &Memory{ID: "m2", Content: "User dislikes Java", CreatedAt: now}, Score: 0.8},
		{Memory: &Memory{ID: "m3", Content: "User budget is 50k", CreatedAt: now}, Score: 0.7},
	}
}

// ---------------------------------------------------------------------------
// NewMemoryFilter factory
// ---------------------------------------------------------------------------

func TestNewMemoryFilter_DefaultIsHeuristic(t *testing.T) {
	f := NewMemoryFilter(config.MemoryReflectionConfig{}, nil)
	require.NotNil(t, f)
	_, isHeuristic := f.(*ReflectionGate)
	assert.True(t, isHeuristic, "default algorithm should produce *ReflectionGate")
}

func TestNewMemoryFilter_ExplicitHeuristic(t *testing.T) {
	f := NewMemoryFilter(config.MemoryReflectionConfig{Algorithm: "heuristic"}, nil)
	require.NotNil(t, f)
	_, isHeuristic := f.(*ReflectionGate)
	assert.True(t, isHeuristic)
}

func TestNewMemoryFilter_ExplicitNoop(t *testing.T) {
	f := NewMemoryFilter(config.MemoryReflectionConfig{Algorithm: "noop"}, nil)
	require.NotNil(t, f)
	_, isNoop := f.(*NoopFilter)
	assert.True(t, isNoop, "noop algorithm should produce *NoopFilter")
}

func TestNewMemoryFilter_UnknownFallsBackToHeuristic(t *testing.T) {
	f := NewMemoryFilter(config.MemoryReflectionConfig{Algorithm: "quantum_ai"}, nil)
	require.NotNil(t, f)
	_, isHeuristic := f.(*ReflectionGate)
	assert.True(t, isHeuristic, "unknown algorithm should fall back to heuristic")
}

func TestNewMemoryFilter_DisabledReturnsNoop(t *testing.T) {
	disabled := false
	f := NewMemoryFilter(config.MemoryReflectionConfig{Enabled: &disabled}, nil)
	require.NotNil(t, f)
	_, isNoop := f.(*NoopFilter)
	assert.True(t, isNoop, "disabled config should return NoopFilter")
}

func TestNewMemoryFilter_PerDecisionAlgorithmOverride(t *testing.T) {
	global := config.MemoryReflectionConfig{Algorithm: "heuristic"}
	perDecision := &config.MemoryReflectionConfig{Algorithm: "noop"}
	f := NewMemoryFilter(global, perDecision)
	require.NotNil(t, f)
	_, isNoop := f.(*NoopFilter)
	assert.True(t, isNoop, "per-decision should override global algorithm")
}

// ---------------------------------------------------------------------------
// RegisterFilter / RegisteredFilters
// ---------------------------------------------------------------------------

func TestRegisterFilter_CustomAlgorithm(t *testing.T) {
	var called bool
	RegisterFilter("test_custom", func(_ config.MemoryReflectionConfig, _ *config.MemoryReflectionConfig) MemoryFilter {
		called = true
		return &NoopFilter{}
	})
	defer func() {
		filterMu.Lock()
		delete(filterRegistry, "test_custom")
		filterMu.Unlock()
	}()

	f := NewMemoryFilter(config.MemoryReflectionConfig{Algorithm: "test_custom"}, nil)
	require.NotNil(t, f)
	assert.True(t, called, "custom factory should have been invoked")
}

func TestRegisteredFilters_ContainsBuiltins(t *testing.T) {
	names := RegisteredFilters()
	sort.Strings(names)
	assert.Contains(t, names, "heuristic")
	assert.Contains(t, names, "noop")
}

// ---------------------------------------------------------------------------
// NoopFilter
// ---------------------------------------------------------------------------

func TestNoopFilter_PassesAllThrough(t *testing.T) {
	f := &NoopFilter{}
	mem := sampleMemories()
	result := f.Filter(mem)
	assert.Equal(t, len(mem), len(result))
	assert.Equal(t, mem, result)
}

func TestNoopFilter_NilSlice(t *testing.T) {
	f := &NoopFilter{}
	result := f.Filter(nil)
	assert.Nil(t, result)
}

// ---------------------------------------------------------------------------
// ChainFilter
// ---------------------------------------------------------------------------

type dropFirstFilter struct{}

func (d *dropFirstFilter) Filter(m []*RetrieveResult) []*RetrieveResult {
	if len(m) == 0 {
		return m
	}
	return m[1:]
}

func TestChainFilter_AppliesInOrder(t *testing.T) {
	chain := NewChainFilter(&dropFirstFilter{}, &dropFirstFilter{})
	mem := sampleMemories()
	result := chain.Filter(mem)
	assert.Len(t, result, 1)
	assert.Equal(t, "m3", result[0].Memory.ID)
}

func TestChainFilter_ShortCircuitsOnEmpty(t *testing.T) {
	callCount := 0
	countingFilter := &countFilter{count: &callCount}

	chain := NewChainFilter(&dropAllFilter{}, countingFilter)
	mem := sampleMemories()
	result := chain.Filter(mem)
	assert.Empty(t, result)
	assert.Equal(t, 0, callCount, "second filter should not be called when first returns empty")
}

func TestChainFilter_SkipsNilEntries(t *testing.T) {
	chain := NewChainFilter(nil, &NoopFilter{}, nil)
	assert.Len(t, chain.filters, 1)
}

type dropAllFilter struct{}

func (d *dropAllFilter) Filter(_ []*RetrieveResult) []*RetrieveResult {
	return nil
}

type countFilter struct{ count *int }

func (c *countFilter) Filter(m []*RetrieveResult) []*RetrieveResult {
	*c.count++
	return m
}
