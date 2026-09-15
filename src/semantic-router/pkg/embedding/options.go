package embedding

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"strings"
)

// Options controls established dimension truncation and layer early exit.
// Zero selects the model's complete output / final layer.
type Options struct {
	Dimension int
	Layer     int
}
type TextRequest struct {
	Text    string
	Options Options
}
type ConfigurableProvider interface {
	Provider
	EmbedWithOptions(context.Context, string, Options) ([]float32, error)
}
type ImageProvider interface {
	EmbedImage(context.Context, []byte, int) ([]float32, error)
}
type AudioProvider interface {
	EmbedAudio(context.Context, []float32, int, int, int) ([]float32, error)
}

// Window offsets are UTF-8 bytes in the original text, with End exclusive.
type Window struct {
	Start int
	End   int
}
type WindowProvider interface {
	Windows(context.Context, string, int) ([]Window, error)
}

// Embed applies options at the provider boundary. Remote providers retain their
// configured server-side output semantics; unsupported local options fail.
func Embed(ctx context.Context, provider Provider, text string, options Options) ([]float32, error) {
	if provider == nil {
		return nil, fmt.Errorf("embedding provider was not prepared")
	}
	if advanced, ok := provider.(ConfigurableProvider); ok {
		return advanced.EmbedWithOptions(ctx, text, options)
	}
	return provider.Embed(ctx, text)
}

func Image(ctx context.Context, provider Provider, imageRef string, dimension int) ([]float32, error) {
	p, ok := provider.(ImageProvider)
	if !ok {
		return nil, fmt.Errorf("embedding provider does not support images")
	}
	var payload []byte
	var err error
	if strings.HasPrefix(imageRef, "/") || strings.HasPrefix(imageRef, "./") {
		payload, err = os.ReadFile(imageRef)
	} else {
		if index := strings.Index(imageRef, ";base64,"); index >= 0 {
			imageRef = imageRef[index+8:]
		}
		payload, err = base64.StdEncoding.DecodeString(imageRef)
	}
	if err != nil {
		return nil, fmt.Errorf("decode embedding image: %w", err)
	}
	return p.EmbedImage(ctx, payload, dimension)
}

type providerView struct {
	Provider
	options Options
}

func WithOptions(provider Provider, options Options) Provider {
	if provider == nil {
		return nil
	}
	return &providerView{Provider: provider, options: options}
}

func (p *providerView) Dimension() int {
	if p.options.Dimension > 0 {
		return p.options.Dimension
	}
	return p.Provider.Dimension()
}

func (p *providerView) Embed(ctx context.Context, text string) ([]float32, error) {
	return Embed(ctx, p.Provider, text, p.options)
}

func (p *providerView) EmbedWithOptions(ctx context.Context, text string, options Options) ([]float32, error) {
	return Embed(ctx, p.Provider, text, options)
}

func (p *providerView) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if p.options == (Options{}) {
		return p.Provider.EmbedBatch(ctx, texts)
	}
	result := make([][]float32, len(texts))
	for i, text := range texts {
		value, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result[i] = value
	}
	return result, nil
}

func (p *providerView) EmbedImage(ctx context.Context, data []byte, dim int) ([]float32, error) {
	q, ok := p.Provider.(ImageProvider)
	if !ok {
		return nil, fmt.Errorf("embedding provider does not support images")
	}
	return q.EmbedImage(ctx, data, dim)
}

func (p *providerView) EmbedAudio(ctx context.Context, data []float32, bins, frames, dim int) ([]float32, error) {
	q, ok := p.Provider.(AudioProvider)
	if !ok {
		return nil, fmt.Errorf("embedding provider does not support audio")
	}
	return q.EmbedAudio(ctx, data, bins, frames, dim)
}

func (p *providerView) Windows(ctx context.Context, text string, limit int) ([]Window, error) {
	q, ok := p.Provider.(WindowProvider)
	if !ok {
		return nil, fmt.Errorf("embedding provider does not support token windows")
	}
	return q.Windows(ctx, text, limit)
}

// CacheIdentity names immutable model/adapter semantics for request-local caches.
// It contains no source text or credentials.
type CacheIdentifiable interface{ CacheIdentity() string }

func Identity(provider Provider) string {
	if identified, ok := provider.(CacheIdentifiable); ok {
		return identified.CacheIdentity()
	}
	return fmt.Sprintf("%T:%p", provider, provider)
}
func (p *providerView) CacheIdentity() string { return Identity(p.Provider) }
