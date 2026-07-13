package classification

import (
	"context"
	"fmt"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// openVINOTextEmbedding is replaceable in tests so backend routing can be
// verified without requiring an OpenVINO runtime or model artifact.
var openVINOTextEmbedding = getOpenVINOEmbedding

// initializeOpenVINOTextEmbedding is replaceable in tests so config-to-runtime
// initialization can be proven without loading a native model artifact.
var initializeOpenVINOTextEmbedding = initOpenVINOModel

// initializeCandleTextEmbedding is used when a remote-configured runtime is
// explicitly switched to a local Candle model.
var initializeCandleTextEmbedding = candle_binding.InitEmbeddingModels

// effectiveTextEmbeddingBackend is the single backend-selection seam for all
// provider-capable classification families. The runtime override has highest
// priority. A constructed provider is next because legacy remote configs can
// express the backend through model_type=remote while HNSW defaults still
// materialize backend=candle. Provider-less classifiers then use their
// explicitly configured backend and finally default to Candle.
func effectiveTextEmbeddingBackend(configuredBackend string, provider embedding.Provider) string {
	if override := embeddingBackendOverride(); override != "" {
		return override
	}
	if provider != nil {
		if backend := normalizeTextEmbeddingBackend(provider.Backend()); backend != "" {
			return backend
		}
	}
	if backend := normalizeTextEmbeddingBackend(configuredBackend); backend != "" {
		return backend
	}
	return config.EmbeddingBackendCandle
}

func normalizeTextEmbeddingBackend(backend string) string {
	return strings.ToLower(strings.TrimSpace(backend))
}

func resolveTextEmbeddingRuntimePlan(cfg *config.RouterConfig, requestedModelType string) (embedding.RuntimePlan, error) {
	if cfg == nil {
		return embedding.RuntimePlan{}, fmt.Errorf("embedding runtime config is nil")
	}
	models := cfg.EmbeddingModels
	if normalized := normalizeTextEmbeddingBackend(requestedModelType); normalized != "" {
		models.EmbeddingConfig.ModelType = normalized
	}
	return embedding.ResolveRuntimePlan(
		models,
		embeddingBackendOverride(),
		embedding.ModelTypeOverrideFromEnv(),
	)
}

func initializeConfiguredTextEmbeddingBackend(cfg *config.RouterConfig, plan embedding.RuntimePlan) error {
	plans, err := configuredTextEmbeddingRuntimePlans(cfg, plan)
	if err != nil {
		return err
	}
	return initializeResolvedTextEmbeddingBackend(cfg, plans)
}

// executeTextEmbedding resolves and executes the effective backend together,
// preventing admission and inference from drifting onto different paths.
func executeTextEmbedding(
	ctx context.Context,
	configuredBackend string,
	provider embedding.Provider,
	text string,
	modelType string,
	targetDim int,
) ([]float32, string, error) {
	backend := effectiveTextEmbeddingBackend(configuredBackend, provider)
	if isLocalNativeEmbeddingBackend(backend) && normalizeTextEmbeddingBackend(modelType) == config.EmbeddingModelTypeRemote {
		return nil, backend, fmt.Errorf("local embedding backend %q requires an explicit local model type", backend)
	}
	switch backend {
	case config.EmbeddingBackendOpenAICompatible:
		if provider == nil {
			return nil, backend, fmt.Errorf("embedding provider is required for backend %q", backend)
		}
		embedding, err := provider.Embed(ctx, text)
		return embedding, backend, err
	case config.EmbeddingBackendOpenVINO:
		embedding, err := openVINOTextEmbedding(modelType, text, targetDim)
		return embedding, backend, err
	case config.EmbeddingBackendCandle:
		output, err := getEmbeddingWithModelType(text, modelType, targetDim)
		if err != nil {
			return nil, backend, err
		}
		if output == nil {
			return nil, backend, fmt.Errorf("embedding backend %q returned no output", backend)
		}
		return output.Embedding, backend, nil
	default:
		return nil, backend, fmt.Errorf("unsupported embedding backend %q", backend)
	}
}

func textEmbeddingUsesLocalNativeBackend(configuredBackend string, provider embedding.Provider) bool {
	return isLocalNativeEmbeddingBackend(effectiveTextEmbeddingBackend(configuredBackend, provider))
}

func isLocalNativeEmbeddingBackend(backend string) bool {
	switch normalizeTextEmbeddingBackend(backend) {
	case config.EmbeddingBackendCandle, config.EmbeddingBackendOpenVINO:
		return true
	default:
		return false
	}
}
