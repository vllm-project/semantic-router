package native

import (
	"fmt"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

type ortEmbeddingEngine interface {
	embeddingEngine
	Info() (ort.Info, error)
}

type ortEmbeddingFactory struct {
	prepareOptions  func(ort.Options) (ort.Options, error)
	load            func(ort.Options) (ortEmbeddingEngine, error)
	contentIdentity bool
	layerViews      bool
	modalities      []string
}

var ortTextFactory = ortEmbeddingFactory{
	load: func(options ort.Options) (ortEmbeddingEngine, error) {
		model, err := ort.LoadEmbeddingModel(options)
		if err != nil {
			return nil, err
		}
		return &ortTextEmbeddingEngine{EmbeddingModel: model}, nil
	},
	contentIdentity: true, layerViews: true, modalities: []string{"text"},
}

var ortEmbeddingFactories = map[string]ortEmbeddingFactory{
	"mmbert":           ortTextFactory,
	"mmbert_embedding": ortTextFactory,
	"modernbert":       ortTextFactory,
	"vela_omni": {
		prepareOptions: prepareFixedGPUExecution,
		load: func(options ort.Options) (ortEmbeddingEngine, error) {
			model, err := ort.LoadOmni(options)
			if err != nil {
				return nil, err
			}
			return &ortOmniEmbeddingEngine{OmniModel: model}, nil
		},
		contentIdentity: true,
		modalities:      []string{"text", "image", "audio"},
	},
	"multimodal": {
		load: func(options ort.Options) (ortEmbeddingEngine, error) {
			model, err := ort.LoadMultiModal(options)
			if err != nil {
				return nil, err
			}
			return &ortMultimodalEmbeddingEngine{MultiModalModel: model}, nil
		},
		modalities: []string{"text", "image", "audio"},
	},
}

func resolveORTEmbeddingFactory(adapter string) (ortEmbeddingFactory, error) {
	factory, ok := ortEmbeddingFactories[adapter]
	if !ok {
		return ortEmbeddingFactory{}, fmt.Errorf("%w: unsupported ORT embedding adapter %q", binding.ErrCapability, adapter)
	}
	return factory, nil
}
