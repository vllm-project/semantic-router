package native

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func (r *Runtime) candleEmbedding(ctx context.Context, spec config.ResolvedModelBinding, view embedding.Options) (*preparedEmbedding, error) {
	options := candleOptions(spec)
	revision, err := r.artifactRevision(ctx, options.ModelPath)
	if err != nil {
		return nil, err
	}
	if spec.Binding.Head != "" {
		return nil, fmt.Errorf("%w: Candle embedding does not support a separate classifier head", binding.ErrCapability)
	}
	execution, _ := json.Marshal(options)
	id := binding.ResourceIdentity{Artifact: options.ModelPath, Revision: spec.Deployment.Revision + ":" + revision, Provider: "candle", Device: options.Device, Precision: options.Precision, Execution: "embedding:" + string(execution)}
	budget, gate := resourceAdmission(spec)
	resource, err := r.Pool.Acquire(ctx, id, budget, gate, func(context.Context) (io.Closer, error) {
		model, loadErr := candle.LoadEmbeddingModel(options)
		if loadErr != nil {
			return nil, nativeError(loadErr)
		}
		return &candleEmbeddingEngine{EmbeddingModel: model}, nil
	})
	if err != nil {
		return nil, err
	}
	prepared := &preparedEmbedding{resource: resource, identity: id}
	err = resource.Use(ctx, func(value io.Closer) error {
		engine := value.(*candleEmbeddingEngine)
		info, infoErr := engine.Info()
		if infoErr != nil {
			return infoErr
		}
		capability := binding.Capability{
			Contract: "embedding.v1", Provider: "candle", Device: info.Device, Precision: info.Precision,
			Limits:    binding.Limits{ModelTokens: info.ArchitecturalMaxTokens, TaskTokens: info.ArchitecturalMaxTokens, DeploymentTokens: spec.Deployment.Input.MaxTokens, Overflow: spec.Deployment.Input.Overflow},
			Embedding: candleEmbeddingSemantics(info.ModelType, view.Layer),
		}
		capability.Embedding.Modalities = append([]string(nil), info.Modalities...)
		capability.Embedding.AvailableDimensions = append([]int(nil), info.AvailableDimensions...)
		if info.ModelType == "mmbert" || info.ModelType == "mmbert_embedding" {
			prepared.layers = candleEmbeddingLayers(options.ModelPath)
			prepared.contentIdentity = true
		}
		capability.Embedding.Dimension, infoErr = warmEmbeddingModel(engine, view, nil)
		prepared.capability = capability
		return infoErr
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	return prepared, nil
}

// These semantics describe the concrete adapters instantiated by the Candle
// loader, rather than assumptions about an arbitrary artifact name.
func candleEmbeddingSemantics(modelType string, layer int) *binding.EmbeddingCapability {
	info := &binding.EmbeddingCapability{Layer: layer, Normalization: "l2", Modalities: []string{"text"}}
	switch modelType {
	case "qwen3":
		info.Pooling = "last_token"
	case "bert", "gemma", "gemma3", "modernbert", "mmbert", "mmbert_embedding":
		info.Pooling = "mean"
	case "multimodal":
		info.Pooling = "mean_text;attention_image;mean_audio"
		info.Modalities = []string{"text", "image", "audio"}
	}
	return info
}

// Candle mmBERT owns every encoder layer. Keep the artifact's advertised exit
// layers within its loaded architecture, and include its final output layer.
func candleEmbeddingLayers(modelPath string) []int {
	data, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil
	}
	var architecture struct {
		Layers int `json:"num_hidden_layers"`
	}
	if json.Unmarshal(data, &architecture) != nil || architecture.Layers <= 0 {
		return nil
	}
	seen := map[int]bool{architecture.Layers: true}
	for _, layer := range config.MmBertAvailableLayers(modelPath) {
		if layer > 0 && layer <= architecture.Layers {
			seen[layer] = true
		}
	}
	layers := make([]int, 0, len(seen))
	for layer := range seen {
		layers = append(layers, layer)
	}
	sort.Ints(layers)
	return layers
}
