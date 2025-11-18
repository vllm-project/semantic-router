package extproc

import (
	"fmt"
	"context"
	"github.com/openai/openai-go"
)

type NewChatCompletionClientOptions struct {
	Endpoint string // chat completion endpoint
}

type ChatCompletionClient struct {
	client openai.Client
}

func NewChatCompletionClient(options NewChatCompletionClientOptions) *ChatCompletionClient {
	c := openai.NewClient(option.WithBaseURL(options.Endpoint))
	return &ChatCompletionClient{
		client: c,
	}
}

func (c *ChatCompletionClient) queryModelForLogProbs(openAIRequest *openai.ChatCompletionNewParams, categoryName string, matchedModel string) (*openai.ChatCompletionChoiceLogprobs, error) {
	openAIRequest.Logprobs = openai.Bool(true)
	openAIRequest.TopLogprobs = openai.Int(5)
	openAIRequest.MaxCompletionTokens = openai.Int(5)
	
	ctx := context.Background()

	// call the client
    resp, err := c.client.Chat.Completions.New(ctx, *openAIRequest)
    if err != nil {
		return nil, fmt.Errorf("error calling chat completions: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned")
	}

	return resp.Choices[0].Logprobs, nil
}
