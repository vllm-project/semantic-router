package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"google.golang.org/grpc/resolver/passthrough"
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

	ragStrategy :=  r.getRAGStrategy(categoryName)
	// TODO: Handle Either RagConfig or DefaultStrategy could be empty
	ragDefault := r.Config.RagConfig.DefaultStrategy 

	// If no RAG strategy is set use the default strategy
	if ragStrategy == "" {
		ragStrategy = ragDefault
	}

	if ragStrategy == "never" {
		return false
	} else if ragStrategy == "always" {
		return true
	} else {
		// TODO: Implement adaptive decision
		return true
	}
}