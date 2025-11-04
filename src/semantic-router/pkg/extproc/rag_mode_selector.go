package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectordb"
)

// getRAGStrategy gets the specified RAG strategy for the categoryName
func (r *OpenAIRouter) getRAGStrategy(categoryName string) string {
	ragStrategy := ""
	for _, category := range r.Config.Categories {
		if strings.EqualFold(category.Name, categoryName) {
			// Get the RAG strategy for this category
			if category.RagStrategy != "" {
				ragStrategy = category.RagStrategy
				if category.RagDescription != "" {
					observability.Infof("RAG strategy: Category '%s', RAG strategy '%s', RAG description '%s'",
						categoryName, ragStrategy, category.RagDescription)
				}
				observability.Infof("RAG strategy: Category '%s', RAG strategy '%s'",
					categoryName, ragStrategy)
			} else {
				observability.Infof("Category '%s' has no RAG strategy configured, defaulting to the default RAG strategy", categoryName)
			}
			break
		}
	}
	return ragStrategy
}

// getRAGDecision determines if RAG should be used based on the query and category
func (r *OpenAIRouter) getRAGDecision(query string, categoryName string) bool {
	// Create a strategy to decision map with string keys and bool values

	ragStrategy := r.getRAGStrategy(categoryName)
	// TODO: Handle Either RagConfig or DefaultStrategy could be empty
	ragDefault := r.Config.Rag.DefaultStrategy

	// If no RAG strategy is set use the default strategy
	if ragStrategy == "" {
		ragStrategy = ragDefault
	}

	switch ragStrategy {
	case "never":
		return false
	case "always":
		return true
	case "adaptive":
		// TODO: Implement adaptive decision: return true
		return true
	default:
		return false
	}
}

func (r *OpenAIRouter) getVectorDBForCategory(categoryName string) vectordb.VectorDbBackend {
	for _, category := range r.Config.Categories {
		if category.Name == categoryName {
			// Use category-specific knowledge base
			if len(category.KnowledgeBases) > 0 {
				return r.KnowledgeBases[category.KnowledgeBases[0]]
			}
			break
		}
	}
	return nil
}

// getRAGChunks gets the RAG chunks which are sent to addRAGChunksToRequestBody as 'ragChunks []string'
func (r *OpenAIRouter) getRAGChunks(categoryName string, query string) ([]string, error) {
	vector_db := r.getVectorDBForCategory(categoryName)
	if vector_db == nil {
		return nil, fmt.Errorf("no vector DB backend found for category %s", categoryName)
	}
	return vector_db.Query(query)
}
