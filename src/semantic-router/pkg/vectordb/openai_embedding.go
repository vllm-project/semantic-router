package vectordb

import (
	"context"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

type NewOpenAIEmbeddingServiceOptions struct {
	Endpoint string // Embedding service endpoint
	// API credentials?
}

type OpenAIEmbeddingService struct {
	client openai.EmbeddingService
}

func NewOpenAIEmbeddingService(options NewOpenAIEmbeddingServiceOptions) *OpenAIEmbeddingService {
	c := openai.NewEmbeddingService(option.WithBaseURL(options.Endpoint))
	return &OpenAIEmbeddingService{
		client: c,
	}
}

func (r *OpenAIEmbeddingService) Embed(input string, model string) ([]float64, error) {
	res, err := r.client.New(context.Background(), openai.EmbeddingNewParams{
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: []string{input}},
		Model: model,
	})
	if err != nil {
		observability.Errorf("Error creating embedding: %s \n", err)
		return nil, err
	}
	// Just return the first embedding since the method currently queries only a single input
	return res.Data[0].Embedding, nil
}
