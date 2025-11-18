package extproc

import (
	"fmt"
	"math"
	"strings"

	"github.com/openai/openai-go"
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
func (r *OpenAIRouter) getRAGDecision(query string, categoryName string, matchedModel string, openAIRequest *openai.ChatCompletionNewParams) bool {
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
		decision, err := r.getAdaptiveRAGDecision(query, categoryName, openAIRequest)
		if err == nil {
			return decision
		} else {
			// TODO: Fallback to a better strategy
			return true
		}
	default:
		return false
	}
}

func (r *OpenAIRouter) getAdaptiveRAGDecision(query string, categoryName string, openAIRequest *openai.ChatCompletionNewParams) (bool, error) {
	logProbs, err := r.getChatResponseLogProbs(query, categoryName, matchedModel, openAIRequest)
	if err != nil {
		observability.Errorf("Error getting log probs: %s \n", err)
		return false, err
	}
	tokenLength := len(logProbs.Content)
	if tokenLength == 0 {
		return false, fmt.Errorf("logprobs not available")
	}
	var totalLogProbs float64 = 0
	for _, tokenLogProbs := range logProbs.Content {
		prob := tokenLogProbs.Logprob
		totalLogProbs += prob
	}
	perplexity := math.Exp(-(totalLogProbs / float64(tokenLength)))

	if perplexity >= r.Config.Rag.ConfidenceThreshold {
		return true, nil
	}
	return false, nil
}

func (r *OpenAIRouter) getChatResponseLogProbs(query string, categoryName string, matchedModel string, openAIRequest *openai.ChatCompletionNewParams) (*openai.ChatCompletionChoiceLogprobs, error) {
	// TODO: Implement
	chatCompletionsClient, err := r.getClientForMatchedModel(matchedModel)
	if err != nil {
		return nil, fmt.Errorf("Unable to create chatCompletionsClient")
	}
	
	// send the openAIRequest
	logProbs, err := chatCompletionsClient.queryModelForLogProbs(openAIRequest, categoryName, matchedModel)
	if err != nil {
		return nil, fmt.Errorf("error fetching log probs: %w", err)
	}

	return logProbs, nil
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

func (r *OpenAIRouter) getClientForMatchedModel(matchedModel string) (ChatCompletionClient, error) {
	// Select the best endpoint for this model
	endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(matchedModel)
	var chatCompletionClientOptions NewChatCompletionClientOptions
	if endpointFound {
		chatCompletionClientOptions = NewChatCompletionClientOptions {
			Endpoint: endpointAddress,
		}
	} else {
		return nil, fmt.Errorf("no endpoint found for matched model: %s", matchedModel)
	}
	return *NewChatCompletionClient(chatCompletionClientOptions)
}
