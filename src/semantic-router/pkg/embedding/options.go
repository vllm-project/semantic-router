package embedding

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
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
	EmbedAudio(context.Context, AudioRequest) ([]float32, error)
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
		return nil, fmt.Errorf("%w: embedding provider does not support images", binding.ErrCapability)
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
		return nil, fmt.Errorf("%w: decode embedding image: %w", binding.ErrInvalidInput, err)
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
		return nil, fmt.Errorf("%w: embedding provider does not support images", binding.ErrCapability)
	}
	return q.EmbedImage(ctx, data, dim)
}

func (p *providerView) EmbedAudio(ctx context.Context, request AudioRequest) ([]float32, error) {
	q, ok := p.Provider.(AudioProvider)
	if !ok {
		return nil, fmt.Errorf("%w: embedding provider does not support audio", binding.ErrCapability)
	}
	return q.EmbedAudio(ctx, request)
}

// Windows reports a missing tokenizer as a capability mismatch, the same way a
// prepared provider without local token windows does, so a caller can tell an
// absent tokenizer from a tokenizer that failed.
func (p *providerView) Windows(ctx context.Context, text string, limit int) ([]Window, error) {
	q, ok := p.Provider.(WindowProvider)
	if !ok {
		return nil, fmt.Errorf("%w: embedding provider does not support token windows", binding.ErrCapability)
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

// Option identities are separate from physical resource identities: two views
// may share one model while producing incompatible vectors.
type OptionCacheIdentifiable interface{ CacheIdentityForOptions(Options) string }

func (p *providerView) CacheIdentity() string {
	if identified, ok := p.Provider.(OptionCacheIdentifiable); ok {
		return identified.CacheIdentityForOptions(p.options)
	}
	return fmt.Sprintf("%s:layer=%d:dimension=%d", Identity(p.Provider), p.options.Layer, p.options.Dimension)
}

func (p *providerView) CacheIdentityForOptions(options Options) string {
	if identified, ok := p.Provider.(OptionCacheIdentifiable); ok {
		return identified.CacheIdentityForOptions(options)
	}
	return fmt.Sprintf("%s:layer=%d:dimension=%d", Identity(p.Provider), options.Layer, options.Dimension)
}

func (p *providerView) RepresentationIdentity(options Options, inputPolicy string) (ContentIdentity, error) {
	owned, ok := p.Provider.(RepresentationProvider)
	if !ok {
		return ContentIdentity{}, ErrIdentityUnsupported
	}
	return owned.RepresentationIdentity(options, inputPolicy)
}

// EmbeddingInfo preserves capabilities when a consumer selects an output view.
func (p *providerView) EmbeddingInfo() ModelInfo {
	info := ModelInfo{Backend: p.Backend(), Dimension: p.Dimension()}
	if described, ok := p.Provider.(Described); ok {
		info = described.EmbeddingInfo()
		info.Dimension = p.Dimension()
	}
	return info
}
