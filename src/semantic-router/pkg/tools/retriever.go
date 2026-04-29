package tools

import "context"

// RetrievalResult holds selected tools plus metadata for observability.
type RetrievalResult struct {
	Tools      []ToolSimilarity
	Confidence float32 // top-1 similarity score, 0 if none
	StrategyID string
}

// Retriever is the interface every tool-retrieval strategy must satisfy.
type Retriever interface {
	Retrieve(ctx context.Context, query string, topK int) (RetrievalResult, error)
}

// Registry maps strategy names to Retriever implementations.
type Registry struct {
	strategies map[string]Retriever
}

func NewRegistry() *Registry {
	return &Registry{strategies: make(map[string]Retriever)}
}

func (r *Registry) Register(name string, s Retriever) {
	if s == nil {
		panic("tools: Register called with nil Retriever for strategy " + name)
	}
	r.strategies[name] = s
}

// Get returns the named strategy, falling back to "default" if not found.
func (r *Registry) Get(name string) (Retriever, bool) {
	s, ok := r.strategies[name]
	return s, ok
}
