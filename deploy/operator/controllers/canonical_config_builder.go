package controllers

import (
	"context"

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
