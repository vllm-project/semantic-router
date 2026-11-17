package modelruntime

import (
	"context"
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

// PrepareOwnedEmbeddings prepares a candidate generation's recipe consumers;
// the service-owned cache, memory and ingestion consumers are prepared by
// PrepareOwnedGlobalServiceEmbeddings. Failure releases only the
// candidate's independent binding references.
func PrepareOwnedEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime) (*embedding.Set, error) {
	return prepareEmbeddings(ctx, cfg, runtime, false, embedding.Options{})
}

// prepareEmbeddings prepares one provider per embedding deployment the
// scope's consumers name, at view, keyed by the deployment's name; the
// set's default is embeddings.default.
func prepareEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime, sharedServices bool, view embedding.Options) (*embedding.Set, error) {
	if cfg == nil {
		return nil, fmt.Errorf("embedding configuration is required")
	}
	if runtime == nil {
		runtime = serving.New(nil, nil)
	}
	recipe := cfg.RoutingScope
	if recipe == "" {
		recipe = config.DefaultRecipeName
	}
	requirements := config.EmbeddingRequirements(cfg, sharedServices)
	providers := make(map[string]embedding.Provider)
	var closers []io.Closer
	fail := func(err error) (*embedding.Set, error) {
		_ = embedding.NewSet(providers, cfg.EmbeddingDeployment, closers...).Close()
		return nil, err
	}
	for _, name := range embeddingDeployments(requirements) {
		spec, err := embeddingSpec(cfg, recipe, name, requirements)
		if err != nil {
			return fail(err)
		}
		provider, err := prepareEmbedding(ctx, cfg, runtime, spec, view)
		if err != nil {
			return fail(fmt.Errorf("prepare embedding deployment %s: %w", name, err))
		}
		providers[name] = provider
		closers = append(closers, provider)
	}
	if err := validatePreparedEmbeddings(ctx, requirements, providers); err != nil {
		return fail(err)
	}
	return embedding.NewSet(providers, cfg.EmbeddingDeployment, closers...), nil
}

// embeddingDeployments are the deployments requirements name, in order.
func embeddingDeployments(requirements []config.EmbeddingRequirement) []string {
	var names []string
	for _, requirement := range requirements {
		if !slices.Contains(names, requirement.Deployment) {
			names = append(names, requirement.Deployment)
		}
	}
	sort.Strings(names)
	return names
}

// embeddingSpec is the binding that prepares a deployment for the consumers
// that name it, admitted by the deployment's own admission. Its name lists
// the shared services that use it, or is "embedding" for a recipe's.
func embeddingSpec(cfg *config.RouterConfig, recipe config.RecipeName, name string, requirements []config.EmbeddingRequirement) (config.ResolvedModelBinding, error) {
	if name == "" {
		return config.ResolvedModelBinding{}, fmt.Errorf("global.model_catalog.embeddings.default names no embedding deployment")
	}
	deployment, ok := cfg.ModelDeployments[name]
	if !ok {
		return config.ResolvedModelBinding{}, fmt.Errorf("embedding deployment %q is not declared", name)
	}
	consumer := "embedding"
	if recipe == config.GlobalModelScope {
		var consumers []string
		for _, requirement := range requirements {
			if requirement.Deployment == name && !slices.Contains(consumers, requirement.Consumer) {
				consumers = append(consumers, requirement.Consumer)
			}
		}
		consumer = strings.Join(consumers, "+") + ".embedding"
	}
	return config.ResolvedModelBinding{
		Recipe: recipe, Name: consumer, Contract: "embedding.v1",
		Binding:    config.ModelBinding{Deployment: name},
		Deployment: deployment.WithDefaults(), Admission: cfg.ModelAdmission[name],
	}, nil
}

// embeddingProvider is a prepared provider the set owns.
type embeddingProvider interface {
	embedding.Provider
	io.Closer
}

// prepareEmbedding serves an embedding deployment: a model_runtime one at
// view, or an openai_embeddings one.
func prepareEmbedding(ctx context.Context, cfg *config.RouterConfig, runtime *serving.Runtime, spec config.ResolvedModelBinding, view embedding.Options) (embeddingProvider, error) {
	if spec.Deployment.Provider != config.OpenAIEmbeddingsProvider {
		return runtime.Embedding(ctx, spec, view.Dimension, view.Layer)
	}
	return prepareRemoteEmbedding(ctx, runtime, spec)
}

// prepareRemoteEmbedding serves an openai_embeddings deployment through its
// connection block.
func prepareRemoteEmbedding(ctx context.Context, runtime *serving.Runtime, spec config.ResolvedModelBinding) (*serving.RemoteEmbeddingProvider, error) {
	deployment := spec.Deployment
	return runtime.RemoteEmbedding(ctx, spec, embedding.OpenAICompatibleConfig{
		BaseURL: deployment.Endpoint, Model: deployment.Model,
		TimeoutSeconds: float64(deployment.TimeoutMs) / 1000, MaxRetries: deployment.RetryCount(0),
		MaxResponseBytes: deployment.ResponseLimitOr(0), Dimensions: deployment.DeclaredDimensions(),
		ExpectedDimension: deployment.DeclaredDimensions(), Deployment: &deployment,
	})
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
