package embedding

import (
	"context"
	"fmt"
	"net/http"
	"os"
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

type ProviderOptions struct {
	BackendOverride string
	HTTPClient      *http.Client
}

type FuncProvider struct {
	backend   string
	dimension int
	embed     func(context.Context, string) ([]float32, error)
}

func BackendOverrideFromEnv() string {
	return strings.ToLower(strings.TrimSpace(os.Getenv("EMBEDDING_BACKEND_OVERRIDE")))
}

func NewProviderFromRouterConfig(cfg *config.RouterConfig, options ProviderOptions) (Provider, error) {
	if cfg == nil {
		return nil, fmt.Errorf("embedding provider config is nil")
	}
	return NewProvider(cfg.EmbeddingModels, options)
}

func NewProvider(models config.EmbeddingModels, options ProviderOptions) (Provider, error) {
	backend := resolveBackend(models, options.BackendOverride)
	switch backend {
	case config.EmbeddingBackendOpenAICompatible:
		return NewOpenAICompatibleProvider(openAICompatibleConfigFromModels(models, options.HTTPClient))
	default:
		return nil, fmt.Errorf("unsupported embedding backend %q", backend)
	}
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

func NewEmbeddingFunc(models config.EmbeddingModels, options ProviderOptions) (func(string) ([]float32, error), error) {
	provider, err := NewProvider(models, options)
	if err != nil {
		return nil, err
	}
	return func(text string) ([]float32, error) {
		return provider.Embed(context.Background(), text)
	}, nil
}

func resolveBackend(models config.EmbeddingModels, override string) string {
	if normalized := strings.ToLower(strings.TrimSpace(override)); normalized != "" {
		return normalized
	}
	return models.EmbeddingBackend()
}

func openAICompatibleConfigFromModels(models config.EmbeddingModels, client *http.Client) OpenAICompatibleConfig {
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
		Dimensions:        models.Endpoint.Dimensions,
		ExpectedDimension: expectedDimension,
		HTTPClient:        client,
	}
}
