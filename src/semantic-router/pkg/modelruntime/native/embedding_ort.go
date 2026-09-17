package native

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func (r *Runtime) ortEmbedding(ctx context.Context, spec config.ResolvedModelBinding, view embedding.Options) (*preparedEmbedding, error) {
	options, err := ortOptions(spec)
	if err != nil {
		return nil, err
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
		if spec.Binding.Adapter == "multimodal" {
			model, loadErr := ort.LoadMultiModal(options)
			if loadErr != nil {
				return nil, loadErr
			}
			return &embeddingEngine{multi: model}, nil
		}
		model, loadErr := ort.LoadEmbeddingModel(options)
		if loadErr != nil {
			return nil, loadErr
		}
		return &embeddingEngine{ort: model}, nil
	})
	if err != nil {
		return nil, err
	}
	prepared := &preparedEmbedding{resource: resource, identity: id}
	err = resource.Use(ctx, func(value io.Closer) error {
		engine := value.(*embeddingEngine)
		var info ort.Info
		var infoErr error
		if engine.ort != nil {
			info, infoErr = engine.ort.Info()
		} else {
			info, infoErr = engine.multi.Info()
		}
		if infoErr != nil {
			return infoErr
		}
		if info.NativeDimension <= 0 {
			return fmt.Errorf("embedding instance did not report a native dimension")
		}
		prepared.dimensionContract = &embedding.DimensionContract{
			NativeDimension:     info.NativeDimension,
			SupportedDimensions: append([]int(nil), info.SupportedDimensions...),
		}
		capability, infoErr := ortCapability(spec, info)
		if infoErr != nil {
			return infoErr
		}
		if engine.ort != nil {
			prepared.layers = append([]int(nil), info.AvailableLayers...)
			prepared.contentIdentity = true
		}
		capability.Embedding = &binding.EmbeddingCapability{Layer: view.Layer, Pooling: "graph_defined", Normalization: "l2", Modalities: []string{"text"}}
		if engine.multi != nil {
			capability.Embedding.Modalities = []string{"text", "image", "audio"}
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
