package embedding

import (
	"context"
	"fmt"
	"strings"
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
