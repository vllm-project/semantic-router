package native

import (
	"fmt"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type candleEmbeddingEngine struct{ *candle.EmbeddingModel }

func (e *candleEmbeddingEngine) embed(text string, options embedding.Options) (tasks.EmbeddingResult, error) {
	out, err := e.EmbedAtLayer(text, options.Dimension, options.Layer)
	return tasks.EmbeddingResult{Embedding: out.Values, Input: &tasks.InputUsage{OriginalTokens: out.Input.InputTokens, ProcessedTokens: out.Input.ProcessedTokens, Truncated: out.Input.Truncated}}, nativeError(err)
}

func (e *candleEmbeddingEngine) embedImage(data []byte, dimension int) ([]float32, error) {
	out, err := e.EmbedImage(data, dimension)
	return out.Values, nativeError(err)
}

func (e *candleEmbeddingEngine) embedAudioFeatures(data []float32, bins, frames, dimension int) ([]float32, error) {
	out, err := e.EmbeddingModel.EmbedAudio(data, bins, frames, dimension)
	return out.Values, nativeError(err)
}

func (e *candleEmbeddingEngine) windows(text string, limit int) ([]embedding.Window, error) {
	ranges, err := e.Windows(text, limit)
	windows := make([]embedding.Window, len(ranges))
	for i, r := range ranges {
		windows[i] = embedding.Window{Start: r.Start, End: r.End}
	}
	return windows, nativeError(err)
}

type ortTextEmbeddingEngine struct{ *ort.EmbeddingModel }

func (e *ortTextEmbeddingEngine) embed(text string, options embedding.Options) (tasks.EmbeddingResult, error) {
	out, err := e.Encode(text, options.Layer, options.Dimension)
	return embeddingORTResult(out), nativeError(ortError(err))
}

func (e *ortTextEmbeddingEngine) windows(text string, limit int) ([]embedding.Window, error) {
	ranges, err := e.Windows(text, limit)
	return ortEmbeddingWindows(ranges), ortError(err)
}

type ortMultimodalEmbeddingEngine struct{ *ort.MultiModalModel }

func (e *ortMultimodalEmbeddingEngine) embed(text string, options embedding.Options) (tasks.EmbeddingResult, error) {
	if options.Layer != 0 {
		return tasks.EmbeddingResult{}, fmt.Errorf("%w: ORT multimodal embedding has no layer early exit", binding.ErrCapability)
	}
	out, err := e.EncodeText(text, options.Dimension)
	return embeddingORTResult(out), nativeError(ortError(err))
}

func (e *ortMultimodalEmbeddingEngine) embedImage(data []byte, dimension int) ([]float32, error) {
	out, err := e.EncodeImageBytes(data, dimension)
	return out.Values, nativeError(ortError(err))
}

func (e *ortMultimodalEmbeddingEngine) embedAudioFeatures(data []float32, bins, frames, dimension int) ([]float32, error) {
	out, err := e.EncodeAudio(data, bins, frames, dimension)
	return out.Values, nativeError(ortError(err))
}

func (e *ortMultimodalEmbeddingEngine) windows(text string, limit int) ([]embedding.Window, error) {
	ranges, err := e.Windows(text, limit)
	return ortEmbeddingWindows(ranges), ortError(err)
}

func embeddingORTResult(out ort.EmbeddingResult) tasks.EmbeddingResult {
	result := tasks.EmbeddingResult{Embedding: out.Values}
	if out.Input != nil {
		result.Input = &tasks.InputUsage{OriginalTokens: out.Input.OriginalTokens, ProcessedTokens: out.Input.ProcessedTokens, Truncated: out.Input.Truncated}
	}
	return result
}

func ortEmbeddingWindows(ranges []ort.TextWindow) []embedding.Window {
	windows := make([]embedding.Window, len(ranges))
	for i, r := range ranges {
		windows[i] = embedding.Window{Start: r.Start, End: r.End}
	}
	return windows
}
