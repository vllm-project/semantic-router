package native

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// RemoteEmbedding owns the HTTP connector and admission reference, never the
// external model process. The declared service is warmed before publication.
func (r *Runtime) RemoteEmbedding(ctx context.Context, spec config.ResolvedModelBinding, cfg embedding.OpenAICompatibleConfig) (prepared *EmbeddingProvider, resultErr error) {
	defer func() { observePreparationFailure(spec, resultErr) }()
	if spec.Deployment.Input.MaxTokens > 0 {
		return nil, fmt.Errorf("%w: remote embedding endpoint does not report tokenizer counts", binding.ErrCapability)
	}
	if cfg.APIKeyEnv != "" {
		cfg.APIKey = os.Getenv(cfg.APIKeyEnv)
		if cfg.APIKey == "" {
			return nil, fmt.Errorf("embedding API key environment is unset")
		}
		cfg.APIKeyEnv = ""
	}
	execution, err := remoteEmbeddingExecution(cfg)
	if err != nil {
		return nil, err
	}
	identity := binding.ResourceIdentity{Artifact: cfg.BaseURL, Revision: cfg.Model, Provider: "http", Device: "external", Precision: "external", Execution: execution}
	budget, gate := resourceAdmission(spec)
	resource, err := r.Pool.Acquire(ctx, identity, budget, gate, func(context.Context) (io.Closer, error) {
		provider, loadErr := embedding.NewOpenAICompatibleProvider(cfg)
		if loadErr != nil {
			return nil, loadErr
		}
		return provider, nil
	})
	if err != nil {
		return nil, err
	}
	task, err := r.embeddingTask()
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	var warm []float32
	err = resource.Use(ctx, func(value io.Closer) error {
		var callErr error
		warm, callErr = value.(*embedding.OpenAICompatibleProvider).Embed(ctx, "semantic router embedding warmup")
		if callErr != nil {
			return callErr
		}
		return validateEmbedding(embedding.TextRequest{}, warm)
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	capability := binding.Capability{Contract: "embedding.v1", Provider: "http", Device: "external", Precision: "external", Embedding: &binding.EmbeddingCapability{Dimension: len(warm), Modalities: []string{"text"}}}
	text, err := task.Resolve(taskIdentity(spec), capability, resource, func(ctx context.Context, value io.Closer, input embedding.TextRequest) (tasks.EmbeddingResult, error) {
		vector, callErr := value.(*embedding.OpenAICompatibleProvider).Embed(ctx, input.Text)
		return tasks.EmbeddingResult{Embedding: vector}, callErr
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	key, _ := identity.Key()
	p := &EmbeddingProvider{identity: key, resource: resource, text: text, recipe: string(spec.Recipe), backend: config.EmbeddingBackendOpenAICompatible}
	p.text.Ready()
	p.dimension = len(warm)
	p.info = embedding.ModelInfo{Artifact: cfg.Model, Backend: p.backend, Dimension: p.dimension, Modalities: []string{"text"}}
	return p, nil
}

// HTTP clients may contain functions and transports; serialize only immutable
// connector settings. A caller-supplied client is shared only with itself.
func remoteEmbeddingExecution(cfg embedding.OpenAICompatibleConfig) (string, error) {
	auth := sha256.Sum256([]byte(cfg.APIKey))
	projected := struct {
		Endpoint, Model, Auth, Client                         string
		TimeoutSeconds, Retries, Dimension, ExpectedDimension int
		MaxResponseBytes                                      int64
	}{cfg.BaseURL, cfg.Model, hex.EncodeToString(auth[:]), fmt.Sprintf("%p", cfg.HTTPClient), cfg.TimeoutSeconds, cfg.MaxRetries, cfg.Dimensions, cfg.ExpectedDimension, cfg.MaxResponseBytes}
	encoded, err := json.Marshal(projected)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}
