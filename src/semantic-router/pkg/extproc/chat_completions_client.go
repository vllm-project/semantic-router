package extproc

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
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

func (c *ChatCompletionClient) queryModelForLogProbs(openAIRequest *openai.ChatCompletionNewParams, matchedModel string) (*openai.ChatCompletionChoiceLogprobs, error) {
	observability.Infof("Querying '%s' for log probs", matchedModel)
	openAIRequest.Model = openai.ChatModel(matchedModel)
	openAIRequest.Logprobs = openai.Bool(true)
	openAIRequest.TopLogprobs = openai.Int(5)
	openAIRequest.MaxCompletionTokens = openai.Int(100)

	ctx := context.Background()

	// call the client
	resp, err := c.client.Chat.Completions.New(ctx, *openAIRequest)
	if err != nil {
		return nil, fmt.Errorf("error calling chat completions: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned")
	}

	logprobs := resp.Choices[0].Logprobs
	observability.Debugf("Full logprobs: %+v", logprobs)
	return &logprobs, nil
}

func (c *ChatCompletionClient) queryModel(openAIRequest *openai.ChatCompletionNewParams, matchedModel string) (string, error) {
	observability.Infof("Querying '%s'", matchedModel)
	openAIRequest.Model = openai.ChatModel(matchedModel)

	ctx := context.Background()

	// call the client
	resp, err := c.client.Chat.Completions.New(ctx, *openAIRequest)
	if err != nil {
		return "", fmt.Errorf("error calling chat completions: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices returned")
	}

	return resp.Choices[0].Message.Content, nil
}
