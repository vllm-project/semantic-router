package config

import "fmt"

const (
	RAGRerankerConsumer     = "rag.reranker"
	RelevanceScoresContract = "relevance_scores.v1"
)

// PairScorerSelection is fixed during loading. Zero selects the actual full
// encoder depth/width; the provider validates trained exits and reports both.
type PairScorerSelection struct {
	Layer     int `json:"layer,omitempty" yaml:"layer,omitempty"`
	Dimension int `json:"dimension,omitempty" yaml:"dimension,omitempty"`
}

// RAGRerankConfig opts structured vectorstore hits into neural pair scoring.
// RAG.top_k remains the candidate count. TopK limits the reordered context.
type RAGRerankConfig struct {
	TopK *int `json:"top_k,omitempty" yaml:"top_k,omitempty"`
}

func validateRerankerBinding(decl ModelBinding, deployment ModelDeployment) error {
	if decl.Adapter != "vela_reranker" || deployment.Provider == "http" {
		return fmt.Errorf("reranker requires a local vela_reranker adapter")
	}
	if decl.MappingPath != "" {
		return fmt.Errorf("reranker returns raw relevance logits and has no label mapping")
	}
	if deployment.Provider == "candle" && decl.Head != "" {
		return fmt.Errorf("candle reranker heads are part of its artifact, not a classifier head binding")
	}
	if deployment.Input.Overflow != "reject" {
		return fmt.Errorf("reranker requires reject overflow for complete tokenizer pairs")
	}
	if decl.PairScorer != nil && (decl.PairScorer.Layer < 0 || decl.PairScorer.Dimension < 0) {
		return fmt.Errorf("pair_scorer layer and dimension must be nonnegative")
	}
	return nil
}

func (c *RAGPluginConfig) validateReranker() error {
	if c.Rerank == nil {
		return nil
	}
	if c.Backend != "vectorstore" {
		return fmt.Errorf("neural rerank requires the structured vectorstore backend")
	}
	if c.Rerank.TopK != nil && *c.Rerank.TopK <= 0 {
		return fmt.Errorf("rerank.top_k must be positive")
	}
	candidates := 5
	if c.TopK != nil {
		candidates = *c.TopK
	}
	if c.Rerank.TopK != nil && *c.Rerank.TopK > candidates {
		return fmt.Errorf("rerank.top_k cannot exceed the RAG candidate top_k")
	}
	return nil
}

// NeedsRAGReranker reports an actual enabled plugin in this recipe scope.
func (c *RouterConfig) NeedsRAGReranker() bool {
	for i := range c.Decisions {
		rag := c.Decisions[i].GetRAGConfig()
		if rag != nil && rag.Enabled && rag.Rerank != nil {
			return true
		}
	}
	return false
}
