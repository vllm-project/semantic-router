package native

import (
	"context"
	"encoding/json"
	"io"
	"path/filepath"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func (r *Runtime) ortEmbedding(ctx context.Context, spec config.ResolvedModelBinding, view embedding.Options) (*preparedEmbedding, error) {
	factory, err := resolveORTEmbeddingFactory(spec.Binding.Adapter)
	if err != nil {
		return nil, err
	}
	options, err := ortOptions(spec)
	if err != nil {
		return nil, err
	}
	if factory.prepareOptions != nil {
		options, err = factory.prepareOptions(options)
		if err != nil {
			return nil, err
		}
	}
	revision, err := r.artifactRevision(ctx, options.ModelPath)
	if err != nil {
		return nil, err
	}
	headRevision := ""
	if options.ModelFile != "" {
		path := options.ModelFile
		if !filepath.IsAbs(path) {
			path = filepath.Join(options.ModelPath, path)
		}
		headRevision, err = r.artifactRevision(ctx, filepath.Dir(path))
		if err != nil {
			return nil, err
		}
	}
	execution, _ := json.Marshal(struct {
		Options               ort.Options
		HeadRevision, Adapter string
	}{options, headRevision, spec.Binding.Adapter})
	d := spec.Deployment.WithDefaults()
	id := binding.ResourceIdentity{Artifact: options.ModelPath, Revision: spec.Deployment.Revision + ":" + revision, Provider: "ort", Device: d.Device, Precision: options.Precision, Execution: "embedding:" + string(execution)}
	budget, gate := resourceAdmission(spec)
	resource, err := r.Pool.Acquire(ctx, id, budget, gate, func(context.Context) (io.Closer, error) {
		return factory.load(options)
	})
	if err != nil {
		return nil, err
	}
	prepared := &preparedEmbedding{resource: resource, identity: id}
	err = resource.Use(ctx, func(value io.Closer) error {
		engine := value.(ortEmbeddingEngine)
		info, infoErr := engine.Info()
		if infoErr != nil {
			return infoErr
		}
		capability, infoErr := ortCapability(spec, info)
		if infoErr != nil {
			return infoErr
		}
		if factory.layerViews {
			prepared.layers = append([]int(nil), info.AvailableLayers...)
		}
		prepared.contentIdentity = factory.contentIdentity
		capability.Embedding = &binding.EmbeddingCapability{Layer: view.Layer, Pooling: "graph_defined", Normalization: "l2", Modalities: append([]string(nil), factory.modalities...)}
		if len(info.Modalities) > 0 {
			capability.Embedding.Modalities = append([]string(nil), info.Modalities...)
		}
		if info.Pooling != "" {
			capability.Embedding.Pooling = info.Pooling
		}
		if info.Normalization != "" {
			capability.Embedding.Normalization = info.Normalization
		}
		capability.Embedding.AvailableDimensions = append([]int(nil), info.AvailableDimensions...)
		if info.Audio != nil {
			capability.Embedding.Audio = &binding.AudioCapability{
				SampleRates: append([]int(nil), info.Audio.SampleRates...), MaxSampleRate: info.Audio.MaxSampleRate,
				MaxSeconds: info.Audio.MaxSeconds, MaxChannels: info.Audio.MaxChannels, Layout: info.Audio.Layout,
			}
		}
		capability.Embedding.Dimension, infoErr = warmEmbeddingModel(engine, view, prepared.layers)
		prepared.capability = capability
		return infoErr
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	return prepared, nil
}
