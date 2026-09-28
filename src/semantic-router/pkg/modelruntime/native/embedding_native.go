package native

import (
	"context"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// Each resource owns one engine. Optional operations are capabilities, so new
// implementations do not add backend branches to request dispatch.
type embeddingEngine interface {
	io.Closer
	embed(string, embedding.Options) (tasks.EmbeddingResult, error)
}
type imageEmbeddingEngine interface {
	embedImage([]byte, int) ([]float32, error)
}
type audioFeatureEmbeddingEngine interface {
	embedAudioFeatures([]float32, int, int, int) ([]float32, error)
}
type audioEmbeddingEngine interface {
	embedAudio(embedding.AudioRequest) ([]float32, error)
}
type windowEmbeddingEngine interface {
	windows(string, int) ([]embedding.Window, error)
}
type descriptorEmbeddingEngine interface {
	RuntimeDescriptor(int, int) (string, error)
}

func (p *EmbeddingProvider) EmbedImage(ctx context.Context, data []byte, dimension int) ([]float32, error) {
	return p.mediaEmbedding(ctx, dimension, func(value io.Closer) ([]float32, error) {
		engine, ok := value.(imageEmbeddingEngine)
		if !ok {
			return nil, fmt.Errorf("%w: embedding provider has no image encoder", binding.ErrCapability)
		}
		return engine.embedImage(data, dimension)
	})
}

// EmbedAudioFeatures accepts model-specific mel features, never raw PCM.
func (p *EmbeddingProvider) EmbedAudioFeatures(ctx context.Context, data []float32, bins, frames, dimension int) ([]float32, error) {
	return p.mediaEmbedding(ctx, dimension, func(value io.Closer) ([]float32, error) {
		engine, ok := value.(audioFeatureEmbeddingEngine)
		if !ok {
			return nil, fmt.Errorf("%w: embedding provider has no audio feature input", binding.ErrCapability)
		}
		return engine.embedAudioFeatures(data, bins, frames, dimension)
	})
}

func (p *EmbeddingProvider) EmbedAudio(ctx context.Context, request embedding.AudioRequest) ([]float32, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	return p.mediaEmbedding(ctx, request.Options.Dimension, func(value io.Closer) ([]float32, error) {
		engine, ok := value.(audioEmbeddingEngine)
		if !ok {
			return nil, fmt.Errorf("%w: embedding provider has no PCM audio input", binding.ErrCapability)
		}
		return engine.embedAudio(request)
	})
}

func (p *EmbeddingProvider) mediaEmbedding(ctx context.Context, dimension int, call func(io.Closer) ([]float32, error)) ([]float32, error) {
	var vector []float32
	err := p.resource.Use(ctx, func(value io.Closer) error {
		var err error
		vector, err = call(value)
		if err != nil {
			return err
		}
		return validateEmbedding(embedding.TextRequest{Options: embedding.Options{Dimension: dimension}}, vector)
	})
	return vector, err
}

func (p *EmbeddingProvider) Windows(ctx context.Context, text string, limit int) ([]embedding.Window, error) {
	var windows []embedding.Window
	err := p.resource.Use(ctx, func(value io.Closer) error {
		engine, ok := value.(windowEmbeddingEngine)
		if !ok {
			return fmt.Errorf("%w: embedding token windows unavailable", binding.ErrCapability)
		}
		var err error
		windows, err = engine.windows(text, limit)
		return err
	})
	return windows, err
}
