// Package tools provides tool selection and filtering capabilities
// for the semantic router.
package tools

import "context"

// EmbeddingRetriever implements ToolRetriever using cosine-similarity search
// against the in-process ToolsDatabase.  It is registered as the "default"
// strategy and requires no external dependencies beyond the database that the
// router already initialises at startup.
type EmbeddingRetriever struct {
	db *ToolsDatabase
}

// NewEmbeddingRetriever returns an EmbeddingRetriever backed by db.
// db must not be nil.
func NewEmbeddingRetriever(db *ToolsDatabase) *EmbeddingRetriever {
	if db == nil {
		panic("tools: NewEmbeddingRetriever called with nil ToolsDatabase")
	}
	return &EmbeddingRetriever{db: db}
}

// Retrieve returns up to in.EffectivePoolSize() tools ranked by embedding
// similarity. HistorySummary, category, and decision fields are ignored.
// The Confidence field of the result is the similarity score of the
// highest-ranked tool, or 0 when no tools are returned.
func (e *EmbeddingRetriever) Retrieve(_ context.Context, in RetrievalInput) (RetrievalResult, error) {
	pool := in.EffectivePoolSize()
	candidates, err := e.db.FindSimilarToolsWithScores(in.Query, pool)
	if err != nil {
		return RetrievalResult{StrategyID: StrategyDefault}, err
	}

	confidence := float32(0)
	if len(candidates) > 0 {
		confidence = candidates[0].Similarity
	}

	return RetrievalResult{
		Tools:      candidates,
		Confidence: confidence,
		StrategyID: StrategyDefault,
	}, nil
}
