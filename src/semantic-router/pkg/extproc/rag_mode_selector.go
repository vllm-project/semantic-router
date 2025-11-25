package extproc

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
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
		observability.Infof("Category '%s' has Adaptive RAG strategy configured", categoryName)
		defaultDecision := false // TODO: Get default from configs
		decision := defaultDecision
		var decisionErr error

		switch r.Config.Rag.DecisionConfig.DecisionMechanism {
		case "logprobs":
			decision, decisionErr = r.getAdaptiveRAGDecisionFromLogProbs(matchedModel, openAIRequest)
		case "reflect":
			decision, decisionErr = r.getAdaptiveRAGDecisionByReflection(matchedModel, openAIRequest)
		case "classify":
			// TODO: Implement. Use a classifier on the query to identify if it requires RAG.
			_ = query // We will use query soon, but do this for now to get rid of the unused warning
		default:
			return defaultDecision
		}

		if decisionErr != nil {
			return decision
		} else {
			return defaultDecision
		}
	default:
		return false
	}
}

func (r *OpenAIRouter) getAdaptiveRAGDecisionFromLogProbs(matchedModel string, openAIRequest *openai.ChatCompletionNewParams) (bool, error) {
	observability.Infof("Getting adaptive RAG Decision")
	logProbs, err := r.getChatResponseLogProbs(matchedModel, openAIRequest)
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
	observability.Infof("Perplexity Score: '%s' ", perplexity)

	// if perplexity >= r.Config.Rag.DecisionConfig.ConfidenceThreshold {
	// 	return true, nil
	// }

	return false, nil
}

func (r *OpenAIRouter) getChatResponseLogProbs(matchedModel string, openAIRequest *openai.ChatCompletionNewParams) (*openai.ChatCompletionChoiceLogprobs, error) {
	observability.Infof("Getting chat response log probs")
	chatCompletionsClient, err := r.getClientForMatchedModel(matchedModel)
	if err != nil {
		return nil, fmt.Errorf("unable to create chatCompletionsClient")
	}

	// send the openAIRequest
	logProbs, err := chatCompletionsClient.queryModelForLogProbs(openAIRequest, matchedModel)
	if err != nil {
		return nil, fmt.Errorf("error fetching log probs: %w", err)
	}

	return logProbs, nil
}

func (r *OpenAIRouter) getAdaptiveRAGDecisionByReflection(matchedModel string, openAIRequest *openai.ChatCompletionNewParams) (bool, error) {
	observability.Infof("Getting RAG decision through reflection")

	chatCompletionsClient, err := r.getClientForMatchedModel(matchedModel)
	if err != nil {
		return false, fmt.Errorf("unable to create chatCompletionsClient")
	}

	ragDecisionTemplatePath := r.Config.Rag.DecisionConfig.RagDecisionTemplatePath
	template, err := loadRAGDecisionTemplate(ragDecisionTemplatePath)
	if err != nil {
		return false, fmt.Errorf("failed to load the RAG decision template")
	}

	lastUserMessage := getLastUserMessage(openAIRequest.Messages)
	if lastUserMessage == nil {
		observability.Debugf("No user message found")
		return false, fmt.Errorf("no user message found")
	}

	updateMessageContentWithRAGDecisionTemplate(&lastUserMessage.Content, template)

	resp, err := chatCompletionsClient.queryModel(openAIRequest, matchedModel)
	if err != nil {
		return false, fmt.Errorf("error fetching log probs: %w", err)
	}

	respLower := strings.ToLower(resp)
	yes := "yes"
	no := "no"
	hasYes := strings.Contains(respLower, yes)
	hasNo := strings.Contains(respLower, no)

	if hasYes && !hasNo {
		// Yes implies the model has sufficient information and does not need RAG
		observability.Debugf("Model does not require RAG")
		return false, nil
	} else if hasNo && !hasYes {
		// No implies the model does not ave sufficient information and needs RAG
		observability.Debugf("Model requires RAG")
		return true, nil
	}
	return false, fmt.Errorf("unclear RAG decision response through reflection")
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

func (r *OpenAIRouter) getClientForMatchedModel(matchedModel string) (*ChatCompletionClient, error) {
	// Select the best endpoint for this model
	endpointAddress, endpointFound := r.Config.SelectBestEndpointAddressForModel(matchedModel)
	var chatCompletionClientOptions NewChatCompletionClientOptions
	if endpointFound {
		endpointAddress := normalizeURL(endpointAddress)
		chatCompletionClientOptions = NewChatCompletionClientOptions{
			Endpoint: endpointAddress,
		}
	} else {
		return nil, fmt.Errorf("no endpoint found for matched model: %s", matchedModel)
	}
	return NewChatCompletionClient(chatCompletionClientOptions), nil
}

func normalizeURL(u string) string {
	if strings.HasPrefix(u, "http://") || strings.HasPrefix(u, "https://") {
		return u + "/v1"
	}
	return "http://" + u + "/v1"
}

func getLastUserMessage(messages []openai.ChatCompletionMessageParamUnion) *openai.ChatCompletionUserMessageParam {
	for i := len(messages) - 1; i >= 0; i-- {
		if !param.IsOmitted(messages[i].OfUser) {
			return messages[i].OfUser
		}
	}
	return nil
}

func loadRAGDecisionTemplate(filePath string) (string, error) {
	templateOnce.Do(func() {
		content, err := os.ReadFile(filePath)
		if err != nil {
			templateFileErr = err
			return
		}
		ragTemplate = string(content)
	})
	return ragTemplate, templateFileErr
}

func updateMessageContentWithRAGDecisionTemplate(content *openai.ChatCompletionUserMessageParamContentUnion, template string) {
	if !param.IsOmitted(content.OfString) {
		// Add to the beginning of the string
		messageContent := content.OfString.String()
		decisionMessage := fmt.Sprintf(template, messageContent)
		content.OfString = param.NewOpt(decisionMessage)
	} else if !param.IsOmitted(content.OfArrayOfContentParts) {
		// Add to beginning of array
		if len(content.OfArrayOfContentParts) > 0 && !param.IsOmitted(content.OfArrayOfContentParts[0].OfText) {
			messageContent := content.OfArrayOfContentParts[0].OfText.Text
			decisionMessage := fmt.Sprintf(template, messageContent)
			content.OfArrayOfContentParts[0].OfText = &openai.ChatCompletionContentPartTextParam{
				Text: decisionMessage,
			}
		}
	}
}
