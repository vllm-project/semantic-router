package native

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// EmbeddingProvider owns one binding reference to a shared physical model.
// Calls hold the resource through admission and non-preemptible native work.
type EmbeddingProvider struct {
	descriptorMu    sync.Mutex
	descriptors     map[embedding.Options]string
	contentIdentity bool
	executionPolicy string
	info            embedding.ModelInfo
	identity        string
	resource        *binding.Resource
	text            *binding.Resolved[embedding.TextRequest, tasks.EmbeddingResult]
	recipe          string
	backend         string
	dimension       int
	options         embedding.Options
	contract        *embedding.DimensionContract
}

func validateEmbeddingResult(input embedding.TextRequest, result tasks.EmbeddingResult) error {
	return validateEmbedding(input, result.Embedding)
}

func validateEmbedding(input embedding.TextRequest, vector []float32) error {
	if len(vector) == 0 {
		return fmt.Errorf("embedding vector is empty")
	}
	if input.Options.Dimension > 0 && len(vector) != input.Options.Dimension {
		return fmt.Errorf("embedding dimension %d differs from requested %d", len(vector), input.Options.Dimension)
	}
	for _, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("embedding vector is not finite")
		}
	}
	return nil
}

func (r *Runtime) embeddingTask() (*binding.Task[embedding.TextRequest, tasks.EmbeddingResult], error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	task, err := binding.Lookup[embedding.TextRequest, tasks.EmbeddingResult](r.registry, "embedding.v1")
	if err == nil {
		return task, nil
	}
	return binding.Register(r.registry, "embedding.v1", func(input embedding.TextRequest) error {
		if input.Options.Dimension < 0 || input.Options.Layer < 0 {
			return fmt.Errorf("embedding dimension and layer must be nonnegative")
		}
		return validateText(input.Text)
	}, validateEmbeddingResult)
}

func (r *Runtime) Embedding(ctx context.Context, spec config.ResolvedModelBinding, dimension, layer int) (prepared *EmbeddingProvider, resultErr error) {
	defer func() { observePreparationFailure(spec, resultErr) }()
	view := embedding.Options{Dimension: dimension, Layer: layer}
	var model *preparedEmbedding
	var err error
	switch spec.Deployment.Provider {
	case "candle":
		model, err = r.candleEmbedding(ctx, spec, view)
	case "ort":
		model, err = r.ortEmbedding(ctx, spec, view)
	default:
		err = fmt.Errorf("%w: native embedding provider %q", binding.ErrCapability, spec.Deployment.Provider)
	}
	if err != nil {
		return nil, err
	}
	resource, id, capability := model.resource, model.identity, model.capability
	layers, contentIdentity := model.layers, model.contentIdentity
	task, err := r.embeddingTask()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	text, err := task.Resolve(taskIdentity(spec), capability, resource, func(_ context.Context, value io.Closer, input embedding.TextRequest) (tasks.EmbeddingResult, error) {
		return value.(*embeddingEngine).embed(input.Text, input.Options)
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	key, _ := id.Key()
	provider := &EmbeddingProvider{
		identity: key,
		resource: resource,
		text:     text,
		recipe:   string(spec.Recipe),
		backend:  spec.Deployment.Provider,
		options:  embedding.Options{Dimension: dimension, Layer: layer},
		contract: model.dimensionContract,
	}
	provider.contentIdentity = contentIdentity
	provider.descriptors = make(map[embedding.Options]string)
	policy, _ := json.Marshal(struct {
		Precision string
		MaxTokens int
		Overflow  string
	}{
		capability.Precision, capability.Limits.EffectiveTokens(), capability.Limits.Overflow,
	})
	provider.executionPolicy = string(policy)
	if contentIdentity {
		identity, identityErr := provider.RepresentationIdentity(provider.options, "embedding-request-cache-v1")
		if identityErr != nil {
			_ = provider.Close()
			return nil, identityErr
		}
		provider.identity = identity.Fingerprint
	}
	provider.text.Ready()
	provider.dimension = capability.Embedding.Dimension
	provider.info = embedding.ModelInfo{Layers: layers, Artifact: id.Artifact, Backend: provider.backend, Dimension: provider.dimension, MaxTokens: capability.Limits.EffectiveTokens(), Pooling: capability.Embedding.Pooling, Normalization: capability.Embedding.Normalization, Modalities: append([]string(nil), capability.Embedding.Modalities...)}
	return provider, nil
}

func (p *EmbeddingProvider) Close() error    { return p.text.Close() }
func (p *EmbeddingProvider) Backend() string { return p.backend }
func (p *EmbeddingProvider) Dimension() int  { return p.dimension }

// EmbeddingDimensionContract returns the dimensions declared by the prepared
// model instance. Callers must use this contract instead of maintaining a
// second model-specific dimension table.
func (p *EmbeddingProvider) EmbeddingDimensionContract() (embedding.DimensionContract, error) {
	if p.contract == nil {
		return embedding.DimensionContract{}, fmt.Errorf("embedding provider does not expose a dimension contract")
	}
	contract := *p.contract
	contract.SupportedDimensions = append([]int(nil), p.contract.SupportedDimensions...)
	return contract, nil
}

func (p *EmbeddingProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.EmbedWithOptions(ctx, text, p.options)
}

func (p *EmbeddingProvider) EmbedWithOptions(ctx context.Context, text string, options embedding.Options) ([]float32, error) {
	result, err := p.text.Call(ctx, p.recipe, embedding.TextRequest{Text: text, Options: options})
	return result.Embedding, err
}

func (p *EmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

func (p *EmbeddingProvider) EmbeddingInfo() embedding.ModelInfo {
	info := p.info
	info.Modalities = append([]string(nil), p.info.Modalities...)
	info.Layers = append([]int(nil), p.info.Layers...)
	return info
}

// Provider preparation returns an already warmed model and its effective
// capabilities. Publication and representation identities stay provider-neutral.
type preparedEmbedding struct {
	resource          *binding.Resource
	identity          binding.ResourceIdentity
	capability        binding.Capability
	layers            []int
	contentIdentity   bool
	dimensionContract *embedding.DimensionContract
}

// ORT exits own separate graphs. Every advertised graph must execute before
// publication, while a Candle backbone only needs its selected view warmed.
func warmEmbeddingModel(engine *embeddingEngine, view embedding.Options, exits []int) (int, error) {
	request := embedding.TextRequest{Text: "warmup", Options: view}
	warm, err := engine.embed(request.Text, request.Options)
	if err != nil {
		return 0, err
	}
	if err = validateEmbeddingResult(request, warm); err != nil {
		return 0, fmt.Errorf("%w: %w", binding.ErrInvalidResult, err)
	}
	dimension := len(warm.Embedding)
	for _, exit := range exits {
		if exit == view.Layer {
			continue
		}
		request.Options.Layer = exit
		output, err := engine.embed(request.Text, request.Options)
		if err != nil {
			return 0, fmt.Errorf("prepare embedding layer %d: %w", exit, err)
		}
		if err = validateEmbeddingResult(request, output); err != nil {
			return 0, fmt.Errorf("%w: embedding layer %d: %w", binding.ErrInvalidResult, exit, err)
		}
		if len(output.Embedding) != dimension {
			return 0, fmt.Errorf("%w: embedding layer %d has a different output dimension", binding.ErrInvalidResult, exit)
		}
	}
	return dimension, nil
}
