package tools

import "context"

// RetrievalResult holds selected tools plus metadata for observability.
type RetrievalResult struct {
	Tools      []ToolSimilarity
	Confidence float32 // top-1 similarity score, 0 if none
	StrategyID string
}

// ToolRetriever is the pluggable interface for tool-candidate retrieval.
// Implementations are registered by name on Registry and selected from
// plugin config (e.g. tools strategy on a matched decision).
type ToolRetriever interface {
	Retrieve(ctx context.Context, in RetrievalInput) (RetrievalResult, error)
}

// Retriever is a type alias for ToolRetriever.
type Retriever = ToolRetriever

// Registry maps strategy names to ToolRetriever implementations.
type Registry struct {
	strategies map[string]ToolRetriever
}

func NewRegistry() *Registry {
	return &Registry{strategies: make(map[string]ToolRetriever)}
}

// NewDefaultRegistry returns a registry with the embedding-similarity strategy
// registered as StrategyDefault.
func NewDefaultRegistry(db *ToolsDatabase) *Registry {
	r := NewRegistry()
	r.Register(StrategyDefault, NewEmbeddingRetriever(db))
	return r
}

func (r *Registry) Register(name string, s ToolRetriever) {
	if s == nil {
		panic("tools: Register called with nil ToolRetriever for strategy " + name)
	}
	r.strategies[name] = s
}

// Get returns the named strategy.
func (r *Registry) Get(name string) (ToolRetriever, bool) {
	s, ok := r.strategies[name]
	return s, ok
}
