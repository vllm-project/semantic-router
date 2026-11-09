package modelruntime

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

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
	cfg = config.WithModelResources(cfg)
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
	if hasExplicit && explicit.Deployment.Provider == config.OpenAIEmbeddingsProvider && cfg.UsesRemoteEmbeddings() {
		// A remote default embedding model serves every text embedding model
		// keyword, as embeddings.semantic.endpoint did. embeddings.multimodal
		// names the multimodal model, which the deprecated endpoint served too.
		ownMultimodal := !explicit.Deployment.Translated() && needed["multimodal"]
		for _, requirement := range requirements {
			if ownMultimodal && requirement.Model == "multimodal" {
				continue
			}
			if err := remoteEmbeddingRequirement(requirement); err != nil {
				return fail(err)
			}
		}
		provider, err := prepareRemoteEmbedding(ctx, cfg, runtime, explicit)
		if err != nil {
			return fail(err)
		}
		closers = append(closers, provider)
		for model := range needed {
			providers[model] = provider
		}
		providers[primary] = provider
		if ownMultimodal {
			spec, err := implicitEmbeddingSpec(cfg, recipe, "multimodal")
			if err != nil {
				return fail(err)
			}
			if recipe == config.GlobalModelScope {
				spec.Name = globalEmbeddingConsumerName(cfg, "multimodal", primary)
			}
			multimodal, err := runtime.Embedding(ctx, spec, 0, 0)
			if err != nil {
				return fail(fmt.Errorf("prepare multimodal embedding: %w", err))
			}
			providers["multimodal"] = multimodal
			closers = append(closers, multimodal)
		}
		if err := validatePreparedEmbeddings(ctx, requirements, providers); err != nil {
			return fail(err)
		}
		return embedding.NewSet(providers, primary, closers...), nil
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
// model_runtime deployment, or an openai_embeddings one.
func prepareExplicitEmbedding(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime, spec config.ResolvedModelBinding, view embedding.Options) (embeddingProvider, error) {
	if spec.Deployment.Provider != config.OpenAIEmbeddingsProvider {
		return runtime.Embedding(ctx, spec, view.Dimension, view.Layer)
	}
	return prepareRemoteEmbedding(ctx, cfg, runtime, spec)
}

// prepareRemoteEmbedding serves an openai_embeddings deployment through its
// connection block.
func prepareRemoteEmbedding(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime, spec config.ResolvedModelBinding) (*serving.RemoteEmbeddingProvider, error) {
	deployment := spec.Deployment
	if _, keyed := cfg.ModelAdmission[spec.Binding.Deployment]; !keyed {
		if admission, legacy := cfg.ModelAdmission["embedding:remote"]; legacy {
			spec.Admission = admission
		}
	}
	return runtime.RemoteEmbedding(ctx, spec, embedding.OpenAICompatibleConfig{
		BaseURL: deployment.Endpoint, Model: deployment.Model,
		TimeoutSeconds: float64(deployment.TimeoutMs) / 1000, MaxRetries: deployment.RetryCount(0),
		MaxResponseBytes: deployment.ResponseLimitOr(0), Dimensions: deployment.DeclaredDimensions(),
		ExpectedDimension: cfg.EmbeddingConfig.TargetDimension, Deployment: &deployment,
	})
}

// implicitEmbeddingSpec is the deployment of an embedding model keyword
// (config.RouterConfig.EmbeddingModelDeployment), at the default exact
// profile. Concurrent requests still share forwards: exact batches queued
// requests together on a model that loads batch-invariant.
func implicitEmbeddingSpec(cfg *config.RouterConfig, recipe config.RecipeName, model string) (config.ResolvedModelBinding, error) {
	name, deployment, err := cfg.EmbeddingModelDeployment(model)
	if err != nil {
		return config.ResolvedModelBinding{}, err
	}
	return config.ResolvedModelBinding{
		Recipe: recipe, Name: "embedding",
		Binding:    config.ModelBinding{Deployment: name, Contract: "embedding.v1"},
		Deployment: deployment, Admission: cfg.EmbeddingAdmission(name, model),
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
