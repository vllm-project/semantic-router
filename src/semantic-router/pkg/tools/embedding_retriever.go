// Package tools provides tool selection and filtering capabilities
// for the semantic router.
package tools

import "context"

// EmbeddingRetriever implements Retriever using cosine-similarity search
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

// Retrieve returns up to topK tools ranked by embedding similarity.
// The Confidence field of the result is the similarity score of the
// highest-ranked tool, or 0 when no tools are returned.
func (e *EmbeddingRetriever) Retrieve(_ context.Context, query string, topK int) (RetrievalResult, error) {
	candidates, err := e.db.FindSimilarToolsWithScores(query, topK)
	if err != nil {
		return RetrievalResult{StrategyID: "default"}, err
	}

	confidence := float32(0)
	if len(candidates) > 0 {
		confidence = candidates[0].Similarity
	}

	return RetrievalResult{
		Tools:      candidates,
		Confidence: confidence,
		StrategyID: "default",
	}, nil
}
