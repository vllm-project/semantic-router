package modelruntime

import (
	"context"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

// embeddingModels are the embedding model types the model runtime serves,
// with the configuration path that names each one's package.
var embeddingModels = map[string]func(*config.RouterConfig) string{
	"mmbert":     func(cfg *config.RouterConfig) string { return cfg.MmBertModelPath },
	"qwen3":      func(cfg *config.RouterConfig) string { return cfg.Qwen3ModelPath },
	"multimodal": func(cfg *config.RouterConfig) string { return cfg.MultiModalModelPath },
}

// PrepareOwnedEmbeddings prepares a candidate generation's recipe consumers;
// the service-owned cache, tools, memory and ingestion consumers are prepared
// by PrepareOwnedGlobalServiceEmbeddings. Failure releases only the
// candidate's independent binding references.
func PrepareOwnedEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime) (*embedding.Set, error) {
	return prepareEmbeddings(ctx, cfg, runtime, false, embedding.Options{})
}

func prepareEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime, sharedServices bool, view embedding.Options) (*embedding.Set, error) {
	if cfg == nil {
		return nil, fmt.Errorf("embedding configuration is required")
	}
	if runtime == nil {
		runtime = serving.New(nil, nil)
	}
	providers := make(map[string]embedding.Provider)
	var closers []io.Closer
	primary := strings.ToLower(strings.TrimSpace(cfg.EmbeddingConfig.ModelType))
	if primary == "" {
		primary = "qwen3"
	}
	fail := func(err error) (*embedding.Set, error) {
		_ = embedding.NewSet(providers, primary, closers...).Close()
		return nil, err
	}
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return fail(err)
	}
	recipe := cfg.RoutingScope
	if recipe == "" {
		recipe = config.DefaultRecipeName
	}
	explicit, hasExplicit := plan.Lookup(recipe, "embedding")
	if recipe == config.GlobalModelScope {
		explicit, hasExplicit = plan.LookupGlobal("embedding")
		explicit.Name = globalEmbeddingConsumerName(cfg, primary, primary)
	}
	if hasExplicit && view == (embedding.Options{}) && cfg.GlobalModelBindings["embedding"].Deployment != "" && primary == "mmbert" {
		configured := cfg.EmbeddingConfig.WithDefaults()
		view = embedding.Options{Dimension: configured.TargetDimension, Layer: configured.TargetLayer}
	}
	needed := config.EmbeddingModelsNeeded(cfg, primary, sharedServices)
	requirements := config.EmbeddingRequirements(cfg, primary, sharedServices)
	if len(needed) == 0 {
		return embedding.NewSet(providers, primary), nil
	}
	if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() && !hasExplicit {
		for _, requirement := range requirements {
			if err := remoteEmbeddingRequirement(requirement); err != nil {
				return fail(err)
			}
		}
		endpoint := cfg.EmbeddingModels.Endpoint
		spec := config.ResolvedModelBinding{Recipe: recipe, Name: "embedding", Binding: config.ModelBinding{Deployment: "embedding:remote", Contract: "embedding.v1", Adapter: "openai_compatible"}, Deployment: config.ModelDeployment{Provider: "http"}, Admission: cfg.ModelAdmission["embedding:remote"]}
		provider, err := runtime.RemoteEmbedding(ctx, spec, embedding.OpenAICompatibleConfig{BaseURL: endpoint.BaseURL, Model: endpoint.Model, APIKeyEnv: endpoint.APIKeyEnv, TimeoutSeconds: endpoint.TimeoutSeconds, MaxRetries: endpoint.MaxRetries, MaxResponseBytes: endpoint.MaxResponseBytes, Dimensions: endpoint.Dimensions, ExpectedDimension: cfg.EmbeddingConfig.TargetDimension})
		if err != nil {
			return fail(err)
		}
		for model := range needed {
			providers[model] = provider
		}
		providers[primary] = provider
		if err := validatePreparedEmbeddings(ctx, requirements, providers); err != nil {
			_ = provider.Close()
			return fail(err)
		}
		return embedding.NewSet(providers, primary, provider), nil
	}
	for _, model := range []string{"qwen3", "gemma", "mmbert", "multimodal", "bert"} {
		if !needed[model] || (hasExplicit && model == primary) {
			continue
		}
		spec, err := implicitEmbeddingSpec(cfg, recipe, model)
		if err != nil {
			return fail(err)
		}
		if recipe == config.GlobalModelScope {
			spec.Name = globalEmbeddingConsumerName(cfg, model, primary)
		}
		var options embedding.Options
		if recipe == config.GlobalModelScope && cfg.SemanticCache.Enabled && model == config.SemanticCacheEmbeddingModel(cfg) {
			options = view
		}
		provider, err := runtime.Embedding(ctx, spec, options.Dimension, options.Layer)
		if err != nil {
			return fail(fmt.Errorf("prepare %s embedding: %w", model, err))
		}
		providers[model] = provider
		closers = append(closers, provider)
	}
	if hasExplicit && needed[primary] {
		provider, err := prepareExplicitEmbedding(ctx, cfg, runtime, explicit, view)
		if err != nil {
			return fail(err)
		}
		providers[primary] = provider
		closers = append(closers, provider)
	}
	if err := validatePreparedEmbeddings(ctx, requirements, providers); err != nil {
		return fail(err)
	}
	return embedding.NewSet(providers, primary, closers...), nil
}

// embeddingProvider is a prepared provider the set owns.
type embeddingProvider interface {
	embedding.Provider
	io.Closer
}

// prepareExplicitEmbedding serves an explicit embedding binding: a
// model_runtime deployment, or an external OpenAI-compatible model.
func prepareExplicitEmbedding(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime, spec config.ResolvedModelBinding, view embedding.Options) (embeddingProvider, error) {
	if spec.Deployment.Provider != "http" {
		return runtime.Embedding(ctx, spec, view.Dimension, view.Layer)
	}
	external := cfg.FindExternalModelByName(spec.Deployment.ExternalModel)
	if external == nil {
		return nil, fmt.Errorf("embedding external_model %q is unavailable", spec.Deployment.ExternalModel)
	}
	address := external.ModelEndpoint.Address
	if external.ModelEndpoint.Port > 0 {
		address = net.JoinHostPort(address, strconv.Itoa(external.ModelEndpoint.Port))
	}
	if !strings.Contains(address, "://") {
		protocol := external.ModelEndpoint.Protocol
		if protocol == "" {
			protocol = "http"
		}
		address = protocol + "://" + address
	}
	return runtime.RemoteEmbedding(ctx, spec, embedding.OpenAICompatibleConfig{BaseURL: address, Model: external.ModelName, APIKey: external.AccessKey, TimeoutSeconds: external.TimeoutSeconds, MaxResponseBytes: external.MaxResponseBytes, ExpectedDimension: cfg.EmbeddingConfig.TargetDimension})
}

// implicitEmbeddingSpec is the module-default deployment of an embedding model
// type: the configured package served by the model runtime as
// "@embedding.<model>", on CPU unless use_cpu is false, at the default exact
// profile. Inputs over budget are truncated, as embeddings always were.
// Concurrent requests still share forwards: exact batches queued requests
// together on a model that loads batch-invariant.
func implicitEmbeddingSpec(cfg *config.RouterConfig, recipe config.RecipeName, model string) (config.ResolvedModelBinding, error) {
	path, served := embeddingModels[model]
	if !served {
		return config.ResolvedModelBinding{}, fmt.Errorf("embedding model %q is not served by the model runtime; use mmbert, qwen3 or multimodal, or an OpenAI-compatible endpoint (vllm-sr config migrate rewrites legacy settings)", model)
	}
	deployment, err := config.ImplicitModelRuntimeDeployment(path(cfg), cfg.EmbeddingModels.UseCPU)
	if err != nil {
		return config.ResolvedModelBinding{}, fmt.Errorf("embedding model %s: %w", model, err)
	}
	deployment.Input.Overflow = "truncate"
	return config.ResolvedModelBinding{
		Recipe: recipe, Name: "embedding",
		Binding:    config.ModelBinding{Deployment: "@embedding." + model, Contract: "embedding.v1"},
		Deployment: deployment, Admission: cfg.ModelAdmission["embedding:"+model],
	}, nil
}

// EmbeddingState describes one prepared set without inferring consumer readiness.
func EmbeddingState(cfg *config.RouterConfig, set *embedding.Set) EmbeddingRuntimeState {
	state := EmbeddingRuntimeState{Embeddings: set, AnyReady: set.Ready()}
	if provider, err := set.Default(); err == nil && provider.Backend() == config.EmbeddingBackendOpenAICompatible {
		state.EmbeddingProvider = remoteEmbeddingProviderProbeStatus(cfg, provider, provider.Dimension(), nil)
		if state.EmbeddingProvider.Model == "" {
			for _, info := range set.Models() {
				state.EmbeddingProvider.Model = info.Artifact
				break
			}
		}
	}
	return state
}
