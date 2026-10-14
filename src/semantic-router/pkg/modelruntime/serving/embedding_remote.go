package serving

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// RemoteEmbeddingProvider serves an embedding binding from an external
// OpenAI-compatible endpoint. It owns the connector and an admission
// reference, never the external model, and shares the vector cache.
type RemoteEmbeddingProvider struct {
	call      *binding.Resolved[embeddingRequest, []tasks.EmbeddingResult]
	recipe    string
	model     string
	dimension int
	space     string
	closed    atomic.Bool
}

// RemoteEmbedding prepares and warms a binding on an OpenAI-compatible endpoint.
func (r *Runtime) RemoteEmbedding(ctx context.Context, spec config.ResolvedModelBinding, cfg embedding.OpenAICompatibleConfig) (_ *RemoteEmbeddingProvider, callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	if spec.Deployment.Input.MaxTokens > 0 {
		return nil, fmt.Errorf("%w: a remote embedding endpoint does not report tokenizer counts", binding.ErrCapability)
	}
	if cfg.APIKeyEnv != "" {
		cfg.APIKey = os.Getenv(cfg.APIKeyEnv)
		if cfg.APIKey == "" {
			return nil, fmt.Errorf("embedding API key environment is unset")
		}
		cfg.APIKeyEnv = ""
	}
	connector, err := remoteConnector(cfg)
	if err != nil {
		return nil, err
	}
	task, err := r.embeddingTask()
	if err != nil {
		return nil, err
	}
	identity := binding.ResourceIdentity{Artifact: cfg.BaseURL, Revision: cfg.Model, Provider: "http", Device: "external", Precision: "external", Execution: connector}
	budget, gate := resourceAdmission(spec)
	resource, err := r.Pool.Acquire(ctx, identity, budget, gate, func(context.Context) (io.Closer, error) {
		return embedding.NewOpenAICompatibleProvider(cfg)
	})
	if err != nil {
		return nil, err
	}
	var warm []float32
	err = resource.Use(ctx, func(value io.Closer) error {
		vectors, callErr := value.(*embedding.OpenAICompatibleProvider).EmbedBatch(ctx, []string{"semantic router embedding warmup"})
		if callErr != nil {
			return callErr
		}
		warm = vectors[0]
		return validateEmbeddingResults(embeddingRequest{Inputs: []modelservice.EmbedInput{{Text: "warmup"}}}, []tasks.EmbeddingResult{{Embedding: warm}})
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	capability := binding.Capability{
		Contract: embeddingContract, Provider: "http", Device: "external", Precision: "external",
		Embedding: &binding.EmbeddingCapability{Dimension: len(warm), Modalities: []string{"text"}},
	}
	call, err := task.Resolve(taskIdentity(spec), capability, resource, func(ctx context.Context, value io.Closer, request embeddingRequest) ([]tasks.EmbeddingResult, error) {
		texts := make([]string, len(request.Inputs))
		for i, input := range request.Inputs {
			if input.Text == "" {
				return nil, fmt.Errorf("%w: a remote embedding endpoint embeds text only", binding.ErrCapability)
			}
			texts[i] = input.Text
		}
		vectors, embedErr := value.(*embedding.OpenAICompatibleProvider).EmbedBatch(ctx, texts)
		if embedErr != nil {
			return nil, embedErr
		}
		results := make([]tasks.EmbeddingResult, len(vectors))
		for i, vector := range vectors {
			results[i] = tasks.EmbeddingResult{Embedding: vector}
		}
		return results, nil
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	call.Ready()
	return &RemoteEmbeddingProvider{call: call, recipe: string(spec.Recipe), model: cfg.Model, dimension: len(warm), space: connector}, nil
}

// remoteConnector fingerprints the connector's immutable settings. The API
// key enters only as a digest; a caller-supplied HTTP client is shared only
// with itself.
func remoteConnector(cfg embedding.OpenAICompatibleConfig) (string, error) {
	auth := sha256.Sum256([]byte(cfg.APIKey))
	encoded, err := json.Marshal(struct {
		Endpoint, Model, Auth, Client                         string
		TimeoutSeconds, Retries, Dimension, ExpectedDimension int
		MaxResponseBytes                                      int64
	}{cfg.BaseURL, cfg.Model, hex.EncodeToString(auth[:]), fmt.Sprintf("%p", cfg.HTTPClient), cfg.TimeoutSeconds, cfg.MaxRetries, cfg.Dimensions, cfg.ExpectedDimension, cfg.MaxResponseBytes})
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

// Close releases the connector reference. A closed provider fails every
// call, cached inputs included.
func (p *RemoteEmbeddingProvider) Close() error {
	p.closed.Store(true)
	return p.call.Close()
}

// Backend names the serving path.
func (p *RemoteEmbeddingProvider) Backend() string { return config.EmbeddingBackendOpenAICompatible }

// Dimension is the endpoint's vector length.
func (p *RemoteEmbeddingProvider) Dimension() int { return p.dimension }

// EmbeddingInfo describes the endpoint's text vectors.
func (p *RemoteEmbeddingProvider) EmbeddingInfo() embedding.ModelInfo {
	return embedding.ModelInfo{Artifact: p.model, Backend: p.Backend(), Dimension: p.dimension, Modalities: []string{"text"}}
}

// CacheIdentity names the endpoint's vector space for request-local caches.
func (p *RemoteEmbeddingProvider) CacheIdentity() string { return "remote:" + p.space }

// Embed embeds one text.
func (p *RemoteEmbeddingProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	vectors, err := p.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// EmbedBatch embeds texts in one endpoint call, answering repeated texts from
// the shared vector cache.
func (p *RemoteEmbeddingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if p.closed.Load() {
		return nil, binding.ErrClosed
	}
	keys := make([]embedding.VectorKey, len(texts))
	for i, text := range texts {
		keys[i] = embedding.NewVectorKey(p.CacheIdentity(), embedding.Options{}, embedding.InputText, []byte(text))
	}
	embedded, err := sharedVectors.Resolve(ctx, keys, func(ctx context.Context, missing []int) ([]embedding.Embedded, error) {
		inputs := make([]modelservice.EmbedInput, len(missing))
		for j, i := range missing {
			inputs[j] = modelservice.EmbedInput{Text: texts[i]}
		}
		results, err := p.call.Call(ctx, p.recipe, embeddingRequest{Inputs: inputs})
		if err != nil {
			return nil, err
		}
		vectors := make([]embedding.Embedded, len(results))
		for i, result := range results {
			vectors[i] = embedding.Embedded{Vector: result.Embedding}
		}
		return vectors, nil
	})
	if err != nil {
		return nil, err
	}
	vectors := make([][]float32, len(embedded))
	for i := range embedded {
		vectors[i] = embedded[i].Vector
	}
	return vectors, nil
}
