package controllers

import (
	"context"
	"fmt"
	"sort"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *SemanticRouterReconciler) buildCanonicalConfig(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) (*routerconfig.CanonicalConfig, error) {
	canonical := &routerconfig.CanonicalConfig{
		Version: "v0.3",
		Listeners: []routerconfig.Listener{
			{
				Name:    "grpc-50051",
				Address: "0.0.0.0",
				Port:    50051,
				Timeout: "300s",
			},
		},
		Routing: routerconfig.CanonicalRouting{
			Signals: routerconfig.CanonicalSignals{
				Domains: []routerconfig.Category{
					{
						CategoryMetadata: routerconfig.CategoryMetadata{
							Name:           "general",
							Description:    "General queries",
							MMLUCategories: []string{"other"},
						},
					},
				},
			},
		},
		Providers: routerconfig.CanonicalProviders{
			Defaults: routerconfig.CanonicalProviderDefaults{},
			Models:   []routerconfig.CanonicalProviderModel{},
		},
		Global: &routerconfig.CanonicalGlobal{
			Services:     routerconfig.CanonicalServiceGlobal{},
			Stores:       routerconfig.CanonicalStoreGlobal{},
			Integrations: routerconfig.CanonicalIntegrationGlobal{},
			ModelCatalog: routerconfig.CanonicalModelCatalog{
				Embeddings: routerconfig.CanonicalEmbeddingModels{},
				System:     routerconfig.CanonicalSystemModels{},
				Modules:    routerconfig.CanonicalModelModules{},
			},
		},
	}

	if err := r.applyDiscoveredBackends(ctx, canonical, sr); err != nil {
		return nil, err
	}
	if err := r.applyOperatorConfigSpec(canonical, sr.Spec.Config); err != nil {
		return nil, err
	}

	return canonical, nil
}

func (r *SemanticRouterReconciler) applyDiscoveredBackends(ctx context.Context, canonical *routerconfig.CanonicalConfig, sr *vllmv1alpha1.SemanticRouter) error {
	if len(sr.Spec.VLLMEndpoints) == 0 {
		return nil
	}

	discoveredModels, err := discoverVLLMBackends(ctx, r.Client, sr.Spec.VLLMEndpoints, sr.Namespace)
	if err != nil {
		return fmt.Errorf("failed to generate vLLM endpoints config: %w", err)
	}
	if len(discoveredModels) == 0 {
		return nil
	}

	modelNames := make([]string, 0, len(discoveredModels))
	for modelName := range discoveredModels {
		modelNames = append(modelNames, modelName)
	}
	sort.Strings(modelNames)

	for index, modelName := range modelNames {
		discovered := discoveredModels[modelName]

		modelCard := routerconfig.RoutingModel{
			Name:  modelName,
			LoRAs: convertLoRAAdapters(discovered.LoRAs),
		}
		canonical.Routing.ModelCards = append(canonical.Routing.ModelCards, modelCard)
		canonical.Providers.Models = append(canonical.Providers.Models, routerconfig.CanonicalProviderModel{
			Name:            modelName,
			ReasoningFamily: discovered.ReasoningFamily,
			BackendRefs:     append([]routerconfig.CanonicalBackendRef(nil), discovered.BackendRefs...),
		})

		if index == 0 {
			canonical.Providers.Defaults.DefaultModel = modelName
		}
	}

	return nil
}

func convertLoRAAdapters(spec []vllmv1alpha1.LoRAAdapterSpec) []routerconfig.LoRAAdapter {
	if len(spec) == 0 {
		return nil
	}
	loras := make([]routerconfig.LoRAAdapter, 0, len(spec))
	for _, adapter := range spec {
		loras = append(loras, routerconfig.LoRAAdapter{
			Name:        adapter.Name,
			Description: adapter.Description,
		})
	}
	return loras
}
