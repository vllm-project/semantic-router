package native

import (
	"context"
	"fmt"
	"io"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type embeddingEngine struct {
	candle *candle.EmbeddingModel
	ort    *ort.EmbeddingModel
	multi  *ort.MultiModalModel
}

func (e *embeddingEngine) Close() error {
	if e.candle != nil {
		return e.candle.Close()
	}
	if e.ort != nil {
		return e.ort.Close()
	}
	return e.multi.Close()
}

func (e *embeddingEngine) embed(text string, options embedding.Options) (tasks.EmbeddingResult, error) {
	if e.candle != nil {
		out, err := e.candle.EmbedAtLayer(text, options.Dimension, options.Layer)
		return tasks.EmbeddingResult{Embedding: out.Values, Input: &tasks.InputUsage{OriginalTokens: out.Input.InputTokens, ProcessedTokens: out.Input.ProcessedTokens, Truncated: out.Input.Truncated}}, nativeError(err)
	}
	if e.ort != nil {
		out, err := e.ort.Encode(text, options.Layer, options.Dimension)
		return embeddingORTResult(out), ortError(err)
	}
	if options.Layer != 0 {
		return tasks.EmbeddingResult{}, fmt.Errorf("%w: ORT multimodal embedding has no layer early exit", binding.ErrCapability)
	}
	out, err := e.multi.EncodeText(text, options.Dimension)
	return embeddingORTResult(out), ortError(err)
}

func embeddingORTResult(out ort.EmbeddingResult) tasks.EmbeddingResult {
	result := tasks.EmbeddingResult{Embedding: out.Values}
	if out.Input != nil {
		result.Input = &tasks.InputUsage{OriginalTokens: out.Input.OriginalTokens, ProcessedTokens: out.Input.ProcessedTokens, Truncated: out.Input.Truncated}
	}
	return result
}

func (p *EmbeddingProvider) EmbedImage(ctx context.Context, data []byte, dimension int) ([]float32, error) {
	var vector []float32
	err := p.resource.Use(ctx, func(value io.Closer) error {
		engine, ok := value.(*embeddingEngine)
		if !ok {
			return fmt.Errorf("%w: embedding provider has no image encoder", binding.ErrCapability)
		}
		var err error
		if engine.candle != nil {
			var out candle.InstanceEmbeddingOutput
			out, err = engine.candle.EmbedImage(data, dimension)
			vector = out.Values
		} else if engine.multi != nil {
			var out ort.EmbeddingResult
			out, err = engine.multi.EncodeImageBytes(data, dimension)
			vector = out.Values
		} else {
			return fmt.Errorf("%w: embedding provider has no image encoder", binding.ErrCapability)
		}
		if err != nil {
			return nativeError(ortError(err))
		}
		return validateEmbedding(embedding.TextRequest{Options: embedding.Options{Dimension: dimension}}, vector)
	})
	return vector, err
}

func (p *EmbeddingProvider) EmbedAudio(ctx context.Context, data []float32, bins, frames, dimension int) ([]float32, error) {
	var vector []float32
	err := p.resource.Use(ctx, func(value io.Closer) error {
		engine, ok := value.(*embeddingEngine)
		if !ok {
			return fmt.Errorf("%w: embedding provider has no audio encoder", binding.ErrCapability)
		}
		var err error
		if engine.candle != nil {
			var out candle.InstanceEmbeddingOutput
			out, err = engine.candle.EmbedAudio(data, bins, frames, dimension)
			vector = out.Values
		} else if engine.multi != nil {
			var out ort.EmbeddingResult
			out, err = engine.multi.EncodeAudio(data, bins, frames, dimension)
			vector = out.Values
		} else {
			return fmt.Errorf("%w: embedding provider has no audio encoder", binding.ErrCapability)
		}
		if err != nil {
			return nativeError(ortError(err))
		}
		return validateEmbedding(embedding.TextRequest{Options: embedding.Options{Dimension: dimension}}, vector)
	})
	return vector, err
}

func (p *EmbeddingProvider) Windows(ctx context.Context, text string, limit int) ([]embedding.Window, error) {
	var windows []embedding.Window
	err := p.resource.Use(ctx, func(value io.Closer) error {
		engine, ok := value.(*embeddingEngine)
		if !ok {
			return fmt.Errorf("%w: remote embedding token windows unavailable", binding.ErrCapability)
		}
		if engine.candle != nil {
			ranges, err := engine.candle.Windows(text, limit)
			for _, r := range ranges {
				windows = append(windows, embedding.Window{Start: r.Start, End: r.End})
			}
			return nativeError(err)
		}
		var ranges []ort.TextWindow
		var err error
		if engine.ort != nil {
			ranges, err = engine.ort.Windows(text, limit)
		} else if engine.multi != nil {
			ranges, err = engine.multi.Windows(text, limit)
		} else {
			return fmt.Errorf("%w: remote embedding token windows unavailable", binding.ErrCapability)
		}
		for _, r := range ranges {
			windows = append(windows, embedding.Window{Start: r.Start, End: r.End})
		}
		return ortError(err)
	})
	return windows, err
}
