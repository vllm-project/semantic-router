package native

import (
	"fmt"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// ortOmniEmbeddingEngine implements the capabilities declared by the loaded
// manifest. The owned native instance validates dimensions and preprocesses
// each modality; the router never reconstructs model math or guesses a variant.
type ortOmniEmbeddingEngine struct{ *ort.OmniModel }

func (e *ortOmniEmbeddingEngine) embed(text string, options embedding.Options) (tasks.EmbeddingResult, error) {
	if options.Layer != 0 {
		return tasks.EmbeddingResult{}, fmt.Errorf("%w: Omni does not support layer exits", binding.ErrCapability)
	}
	out, err := e.EncodeText(text, options.Dimension)
	return embeddingORTResult(out), ortError(err)
}

func (e *ortOmniEmbeddingEngine) embedImage(data []byte, dimension int) ([]float32, error) {
	out, err := e.EncodeImageBytes(data, dimension)
	return out.Values, ortError(err)
}

func (e *ortOmniEmbeddingEngine) embedAudio(request embedding.AudioRequest) ([]float32, error) {
	out, err := e.EncodeAudioPCM(request.PCM, request.SampleRate, request.Channels, request.Options.Dimension)
	return out.Values, ortError(err)
}

func (e *ortOmniEmbeddingEngine) windows(text string, limit int) ([]embedding.Window, error) {
	ranges, err := e.Windows(text, limit)
	return ortEmbeddingWindows(ranges), ortError(err)
}
