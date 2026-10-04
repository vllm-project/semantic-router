package embedding

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Provider computes text embeddings for router runtime features.
type Provider interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
	Dimension() int
	Backend() string
}

type FuncProvider struct {
	backend   string
	dimension int
	embed     func(context.Context, string) ([]float32, error)
}

// NewProvider builds the provider of a remote embedding backend. Local
// embeddings come from the model runtime's prepared set instead.
func NewProvider(models config.EmbeddingModels) (Provider, error) {
	if backend := models.EmbeddingBackend(); backend != config.EmbeddingBackendOpenAICompatible {
		return nil, fmt.Errorf("unsupported embedding backend %q", backend)
	}
	return NewOpenAICompatibleProvider(openAICompatibleConfigFromModels(models))
}

func NewFuncProvider(backend string, dimension int, embed func(context.Context, string) ([]float32, error)) (*FuncProvider, error) {
	backend = strings.ToLower(strings.TrimSpace(backend))
	if backend == "" {
		return nil, fmt.Errorf("embedding backend is required")
	}
	if embed == nil {
		return nil, fmt.Errorf("embedding function is required for backend %q", backend)
	}
	return &FuncProvider{backend: backend, dimension: dimension, embed: embed}, nil
}

func (p *FuncProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.embed(ctx, text)
}

func (p *FuncProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		embedding, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		embeddings[i] = embedding
	}
	return embeddings, nil
}

func (p *FuncProvider) Dimension() int {
	return p.dimension
}

func (p *FuncProvider) Backend() string {
	return p.backend
}

func openAICompatibleConfigFromModels(models config.EmbeddingModels) OpenAICompatibleConfig {
	expectedDimension := models.EmbeddingConfig.TargetDimension
	if models.Endpoint.Dimensions > 0 {
		expectedDimension = models.Endpoint.Dimensions
	}
	return OpenAICompatibleConfig{
		BaseURL:           models.Endpoint.BaseURL,
		Model:             models.Endpoint.Model,
		APIKeyEnv:         models.Endpoint.APIKeyEnv,
		TimeoutSeconds:    models.Endpoint.TimeoutSeconds,
		MaxRetries:        models.Endpoint.MaxRetries,
		MaxResponseBytes:  models.Endpoint.MaxResponseBytes,
		Dimensions:        models.Endpoint.Dimensions,
		ExpectedDimension: expectedDimension,
	}
}
