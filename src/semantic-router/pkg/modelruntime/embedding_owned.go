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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// PrepareOwnedEmbeddings prepares the catalog for a candidate generation.
// Failure releases only the candidate's independent resource references.
func PrepareOwnedEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *native.Runtime) (*embedding.Set, error) {
	return prepareEmbeddings(ctx, cfg, runtime, true)
}

// PrepareOwnedRecipeEmbeddings excludes service-owned cache, tools, memory and
// ingestion resources. A standalone classifier owns only its recipe consumers.
func PrepareOwnedRecipeEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *native.Runtime) (*embedding.Set, error) {
	return prepareEmbeddings(ctx, cfg, runtime, false)
}

func prepareEmbeddings(ctx context.Context, cfg *config.RouterConfig, runtime *native.Runtime, sharedServices bool) (*embedding.Set, error) {
	if cfg == nil {
		return nil, fmt.Errorf("embedding configuration is required")
	}
	if runtime == nil {
		runtime = native.New(nil)
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

	needed := embeddingNeedsForScope(cfg, primary, sharedServices)
	requirements := config.EmbeddingRequirements(cfg, primary, sharedServices)
	if cfg.EmbeddingModels.EmbeddingBackend() == config.EmbeddingBackendOpenVINO && !hasExplicit {
		// Legacy OpenVINO recipe classifiers initialize their own primary.
		// Independent service providers still belong to this owned snapshot.
		delete(needed, primary)
		ownedRequirements := requirements[:0]
		for _, requirement := range requirements {
			if requirement.Model == primary && !requirement.SharedService {
				continue
			}
			ownedRequirements = append(ownedRequirements, requirement)
			if requirement.SharedService {
				needed[requirement.Model] = true
			}
		}
		requirements = ownedRequirements
	}
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

	paths := resolveEmbeddingPaths(cfg)
	models := map[string]string{"qwen3": paths.qwen3, "gemma": paths.gemma, "mmbert": paths.mmBert, "multimodal": paths.multiModal, "bert": paths.bert}
	if semanticCacheNeedsBERT(cfg) || vectorStoreNeedsBERT(cfg) || memoryNeedsBERT(cfg) {
		models["bert"] = resolveBertModelID(cfg.BertModelPath)
	}
	for _, model := range []string{"qwen3", "gemma", "mmbert", "multimodal", "bert"} {
		path := models[model]
		if !needed[model] {
			continue
		}
		if model == "bert" && path == "" {
			path = resolveBertModelID(cfg.BertModelPath)
		}
		if hasExplicit && model == primary {
			continue
		}
		if path == "" {
			return fail(fmt.Errorf("required embedding model %q has no artifact path", model))
		}
		provider, err := runtime.Embedding(ctx, embeddingCatalogSpec(cfg, recipe, model, path), 0, 0)
		if err != nil {
			return fail(fmt.Errorf("prepare %s embedding: %w", model, err))
		}
		providers[model] = provider
		closers = append(closers, provider)
	}
	if hasExplicit && needed[primary] {
		explicit.Deployment.Artifact = config.ResolveModelPath(explicit.Deployment.Artifact)

		var provider *native.EmbeddingProvider
		var err error
		if explicit.Deployment.Provider == "http" {
			external := cfg.FindExternalModelByName(explicit.Deployment.ExternalModel)
			if external == nil {
				return fail(fmt.Errorf("embedding external_model %q is unavailable", explicit.Deployment.ExternalModel))
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
			provider, err = runtime.RemoteEmbedding(ctx, explicit, embedding.OpenAICompatibleConfig{BaseURL: address, Model: external.ModelName, APIKey: external.AccessKey, TimeoutSeconds: external.TimeoutSeconds, MaxResponseBytes: external.MaxResponseBytes, ExpectedDimension: cfg.EmbeddingConfig.TargetDimension})
		} else {
			provider, err = runtime.Embedding(ctx, explicit, 0, 0)
		}

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

func embeddingCatalogSpec(cfg *config.RouterConfig, recipe config.RecipeName, model, path string) config.ResolvedModelBinding {
	provider, device := config.DefaultModelExecution(cfg.EmbeddingModels.UseCPU)
	return config.ResolvedModelBinding{Recipe: recipe, Name: "embedding", Binding: config.ModelBinding{Deployment: "embedding:" + model, Contract: "embedding.v1", Adapter: model}, Deployment: config.ModelDeployment{Artifact: path, Provider: provider, Device: device, Precision: "native", Input: config.ModelInputBudget{Overflow: "truncate"}}, Admission: cfg.ModelAdmission["embedding:"+model]}
}

// EmbeddingState describes the already warmed generation without another call.
func EmbeddingState(cfg *config.RouterConfig, set *embedding.Set) EmbeddingRuntimeState {
	state := EmbeddingRuntimeState{Embeddings: set, AnyReady: set.Ready(), ToolsReady: set.Has("")}
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
